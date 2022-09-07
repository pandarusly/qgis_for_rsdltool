import threading
import queue
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
import torch.nn.functional as F
import collections
from torch.nn.modules.batchnorm import _BatchNorm
from pip import main
import torch
import torch.nn as nn
from ._utils import init_weights

# bn_mom = 0.0003
bn_mom = 0.1

# SynchronizedBatchNorm2d


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple(
    '_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(
                _ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(
                _ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * \
                _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(
            intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * \
            self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * \
            self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


# DFM
"""Implemention of dense fusion module"""


class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                SynchronizedBatchNorm2d(dim_in//2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # print(x1.shape)
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y

# nlfpn


class NL_Block(nn.Module):
    def __init__(self, in_channels):
        super(NL_Block, self).__init__()
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_channels),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        value = self.conv_v(x).view(batch_size, c, -1)
        value = value.permute(0, 2, 1)  # B * (H*W) * value_channels
        key = x.view(batch_size, c, -1)  # B * key_channels * (H*W)
        query = x.view(batch_size, c, -1)
        query = query.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)  # B * (H*W) * (H*W)
        sim_map = (c**-.5) * sim_map  # B * (H*W) * (H*W)
        sim_map = torch.softmax(sim_map, dim=-1)  # B * (H*W) * (H*W)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, c, *x.size()[2:])
        context = self.W(context)

        return context


class NL_FPN(nn.Module):
    """ non-local feature parymid network"""

    def __init__(self, in_dim, reduction=True):
        super(NL_FPN, self).__init__()
        if reduction:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_dim, in_dim//4, kernel_size=1,
                          stride=1, padding=0),
                SynchronizedBatchNorm2d(in_dim//4, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            self.re_reduction = nn.Sequential(
                nn.Conv2d(in_dim//4, in_dim, kernel_size=1,
                          stride=1, padding=0),
                SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            in_dim = in_dim//4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*2, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_dim*2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim*4, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_dim*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_dim, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_dim*4, in_dim*2, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(in_dim*2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.nl3 = NL_Block(in_dim*2)
        self.nl2 = NL_Block(in_dim)
        self.nl1 = NL_Block(in_dim)

        self.downsample_x2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)  # C,H,W
        e2 = self.conv_e2(self.downsample_x2(e1))  # 2C,H/2,W/2
        e3 = self.conv_e3(self.downsample_x2(e2))  # 4C,H/4,W/4

        d3 = self.conv_d3(e3)  # 2C,H/4,W/4
        nl = self.nl3(d3)
        d3 = self.upsample_x2(torch.mul(d3, nl))  # 2C,H/2,W/2
        d2 = self.conv_d2(e2+d3)  # C,H/2,W/2
        nl = self.nl2(d2)
        d2 = self.upsample_x2(torch.mul(d2, nl))  # C,H,W
        d1 = self.conv_d1(e1+d2)
        nl = self.nl1(d1)
        d1 = torch.mul(d1, nl)  # C,H,W
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)

        return d1
# utils


class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(cat, self).__init__()  # parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn_high + in_chn_low, out_chn,
                            kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        # import ipdb
        # ipdb.set_trace()
        if self.do_upsample:
            x = self.upsample(x)

        # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        x = torch.cat((x, y), 1)
        return self.conv2d(x)


class double_conv(torch.nn.Module):
    # params:in_chn(input channel of double conv),out_chn(output channel of double conv)
    def __init__(self, in_chn, out_chn, stride=1, dilation=1):
        super(double_conv, self).__init__()  # parent's init func

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=3,
                            stride=stride, dilation=dilation, padding=dilation),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3,
                            stride=1, padding=1),
            SynchronizedBatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        #x_se = self.avg_pool(x)
        x_se = x.view(x.size(0), x.size(1), -
                      1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, use_se=False, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        first_planes = planes
        outplanes = planes * self.expansion

        self.conv1 = double_conv(inplanes, first_planes)
        self.conv2 = double_conv(
            first_planes, outplanes, stride=stride, dilation=dilation)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = torch.nn.MaxPool2d(
            stride=2, kernel_size=2) if downsample else None
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        residual = out
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)

        return out

# -------------------- main model FCS DED FCCDN


class FCS(torch.nn.Module):
    def __init__(self, num_band, os=16, use_se=False, **kwargs):
        super(FCS, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(
            num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(
            channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(
            channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(
            channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])
        # decoder
        self.decoder3 = cat(
            channel_list[0], channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2 = cat(
            channel_list[1], channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1 = cat(
            channel_list[2], channel_list[3], channel_list[3], upsample=pool_list[2])

        self.df1 = cat(channel_list[3], channel_list[3],
                       channel_list[3], upsample=False)
        self.df2 = cat(channel_list[2], channel_list[2],
                       channel_list[2], upsample=False)
        self.df3 = cat(channel_list[1], channel_list[1],
                       channel_list[1], upsample=False)
        self.df4 = cat(channel_list[0], channel_list[0],
                       channel_list[0], upsample=False)

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[3], 8, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(8, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out = torch.nn.Conv2d(
            8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        e1_1 = self.block1(x[0])
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(x[1])
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        c1 = self.df1(e1_1, e1_2)
        c2 = self.df2(e2_1, e2_2)
        c3 = self.df3(e3_1, e3_2)
        c4 = self.df4(y1, y2)

        y = self.decoder3(c4, c3)
        y = self.decoder2(y, c2)
        y = self.decoder1(y, c1)

        y = self.conv_out(self.upsample_x2(y))
        return [y]


class DED(torch.nn.Module):
    def __init__(self, num_band, os=16, use_se=False, **kwargs):
        super(DED, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(
            num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(
            channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(
            channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(
            channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])

        # center
        # self.center = NL_FPN(channel_list[0], True)

        # decoder
        self.decoder3 = cat(
            channel_list[0], channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2 = cat(
            channel_list[1], channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1 = cat(
            channel_list[2], channel_list[3], channel_list[3], upsample=pool_list[2])

        # self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        # self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        # self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        # self.df4 = DF_Module(channel_list[0], channel_list[0], True)

        self.df1 = cat(channel_list[3], channel_list[3],
                       channel_list[3], upsample=False)
        self.df2 = cat(channel_list[2], channel_list[2],
                       channel_list[2], upsample=False)
        self.df3 = cat(channel_list[1], channel_list[1],
                       channel_list[1], upsample=False)
        self.df4 = cat(channel_list[0], channel_list[0],
                       channel_list[0], upsample=False)

        self.catc3 = cat(
            channel_list[0], channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2 = cat(
            channel_list[1], channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1 = cat(
            channel_list[2], channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[3], 8, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(8, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out = torch.nn.Conv2d(
            8, 1, kernel_size=3, stride=1, padding=1)
        # self.conv_out_class = torch.nn.Conv2d(channel_list[3],1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        e1_1 = self.block1(x[0])
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(x[1])
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        # y1 = self.center(y1)
        # y2 = self.center(y2)
        c = self.df4(y1, y2)

        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)
        c = self.catc1(c, self.df1(y1, y2))
        # y1 = self.conv_out_class(y1)
        # y2 = self.conv_out_class(y2)
        y = self.conv_out(self.upsample_x2(c))
        return [y]


class FCCDN(torch.nn.Module):
    def __init__(self, num_band, os=16, use_se=False, num_class=2, **kwargs):
        super(FCCDN, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(
            num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(
            channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(
            channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(
            channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])

        # center
        self.center = NL_FPN(channel_list[0], True)

        # decoder
        self.decoder3 = cat(
            channel_list[0], channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2 = cat(
            channel_list[1], channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1 = cat(
            channel_list[2], channel_list[3], channel_list[3], upsample=pool_list[2])

        self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        self.df4 = DF_Module(channel_list[0], channel_list[0], True)

        # self.df1 = cat(channel_list[3],channel_list[3], channel_list[3], upsample=False)
        # self.df2 = cat(channel_list[2],channel_list[2], channel_list[2], upsample=False)
        # self.df3 = cat(channel_list[1],channel_list[1], channel_list[1], upsample=False)
        # self.df4 = cat(channel_list[0],channel_list[0], channel_list[0], upsample=False)

        self.catc3 = cat(
            channel_list[0], channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2 = cat(
            channel_list[1], channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1 = cat(
            channel_list[2], channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(channel_list[3], 8, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm2d(8, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out = torch.nn.Conv2d(
            8, num_class, kernel_size=3, stride=1, padding=1)
        self.conv_out_class = torch.nn.Conv2d(
            channel_list[3], num_class, kernel_size=1, stride=1, padding=0)

    def init_weights(self):
        init_weights(self, 'kaiming')

    def forward(self, x1, x2):
        e1_1 = self.block1(x1)
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(x2)
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        y1 = self.center(y1)
        y2 = self.center(y2)

        c = self.df4(y1, y2)

        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)
        c = self.catc1(c, self.df1(y1, y2))
        y1 = self.conv_out_class(y1)
        y2 = self.conv_out_class(y2)
        y = self.conv_out(self.upsample_x2(c))
        return [y, y1, y2]

    def _auxiliary_head_show(self, x1, x2):

        _, pred_a, pred_b = self.forward(x1, x2)

        pred_a = F.interpolate(
            pred_a, x1.shape[2:], scale_factor=None, mode="bilinear", align_corners=False)
        pred_b = F.interpolate(
            pred_b, x1.shape[2:], scale_factor=None, mode="bilinear", align_corners=False)

        return pred_a.argmax(1), pred_b.argmax(1)

    def forward_dummy(self, x1, x2):

        preds = self.forward(x1, x2)

        return preds[0]


model_dict = {
    "FCS": FCS,
    "DED": DED,
    "FCCDN": FCCDN,
}


def GenerateNet(cfg):
    return model_dict[cfg.MODEL_NAME](
        os=cfg.MODEL_OUTPUT_STRIDE,
        num_band=cfg.BAND_NUM,
        use_se=cfg.USE_SE,
    )


if __name__ == '__main__':
    class Config():
        def __init__(self):
            self.MODEL_NAME = 'FCCDN'
            self.MODEL_OUTPUT_STRIDE = 16
            self.BAND_NUM = 3
            self.USE_SE = True

    toy_data1 = torch.rand(2, 3, 256, 256).cuda()
    toy_data2 = torch.rand(2, 3, 256, 256).cuda()

    toy_label1 = torch.randint(0, 2, (2, 1, 256, 256)).cuda()
    toy_label2 = torch.randint(0, 2, (2, 1, 128, 128)).cuda()
    label = [toy_label1.float(), toy_label2.float()]
    cfg = Config()
    model = GenerateNet(cfg)
    model = model.cuda()
    output = model(input)
    for x in output:
        print(x.shape)
    # loss = FCCDN_loss(output, label)
    # print(loss)
