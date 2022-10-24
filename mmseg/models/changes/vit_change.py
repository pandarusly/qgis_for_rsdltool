import collections
import math
from itertools import repeat

import torch
import torch.nn.functional as F
# -------------vvitConvModule
from einops import rearrange, einops

from mmseg.ops import resize


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

from torch import nn, einsum, Tensor


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed3d(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 norm_layer=nn.LayerNorm,
                 input_size=None,
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)

        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.proj = nn.Conv3d(in_channels,
                              embed_dims,
                              kernel_size=(1,) + kernel_size,
                              stride=(1,) + stride,
                              padding=(0,) + padding,
                              dilation=(1,) + dilation
                              )
        # -----
        if norm_layer is not None:
            self.norm = norm_layer(embed_dims)
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x1, x2):
        """
        Args:
            x1 (Tensor): Has shape (B, C, H, W). In most case, C is 3.
            x2 (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """
        """[summary]

        Args:
        x1 [b  c h w ] |  x2 [b  c h w ]

        Returns:
            [type]: [b t (h w) c ] t=2
        """
        if self.adap_padding:
            x1 = self.adap_padding(x1)
            x2 = self.adap_padding(x2)

        x = torch.stack([x1, x2], dim=2)
        # print(x.shape)  # torch.Size([1, 64, 2, 16, 16])
        x = self.proj(x)  # N C T H W
        out_size = (x.shape[3], x.shape[4])
        x = x.transpose(1, 2).flatten(3).transpose(2, 3)  # N C T H W -> N T C (H W) -> N T (H W) C

        if self.norm is not None:
            x = rearrange(x, "N T D C -> N (T D) C")  # 一起做归一化或者 Batch上做
            x = self.norm(x)
            x = rearrange(x, "N (T D) C -> N T D C", T=2)
        return x, out_size


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Sattention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., sr_ratio=1):
        super().__init__()

        project_out = not (heads == 1 and dim_head == dim)
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.sr_ratio = sr_ratio
        # 实现上这里等价于一个卷积层
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)

    def forward(self, x, H, W):
        """
        x: (b, n, d)
        """
        b, n, d, h = *x.shape, self.heads

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        if self.sr_ratio > 1:

            x_ = rearrange(x[:, 1:], 'b n d -> b d n',
                           b=b).reshape(b, d, H, W)

            x_ = rearrange(self.sr(x_).reshape(
                b, d, -1), 'b d n ->b n d')

            # print(x[:, 0].shape)
            x_ = torch.cat([x[:, 0].unsqueeze(1), x_], dim=1)

            kv = self.to_kv(x_).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class STransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., sr_ratio=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Sattention(dim, heads=heads,
                                        dim_head=dim_head, sr_ratio=sr_ratio, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x, H=H, W=W) + x

            x = ff(x) + x

        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


from mmseg.models.builder import CHANGES


from .fusion_head import AsppFusion
@CHANGES.register_module()
class VVITv3(nn.Module):
    def __init__(self, image_size=None, dim=128, patch_size=1, depth=2, heads=3, in_channels=2048, dim_head=256,
                 sr_ratio=2,
                 dropout=0., emb_dropout=0.1, scale_dim=4, pos_norm=nn.LayerNorm):
        """[RSAvit模塊]

        Args:
            x ([b c t h w]): [description]

        Returns:
            [type]: [b c t h w]
        """
        super().__init__()

        self.stride = 1

        self.adap_padding = AdaptivePadding(
            kernel_size=patch_size,
            stride=self.stride,
            padding='corner')

        if image_size:
            self.img_size = to_2tuple(image_size)
            pad_h, pad_w = self.adap_padding.get_pad_shape(self.img_size)
            input_h, input_w = self.img_size
            input_h = input_h + pad_h
            input_w = input_w + pad_w
            self.img_size = (input_h, input_w)

            pos_h = (self.img_size[0] - 1 *
                     (patch_size - 1) - 1) // self.stride + 1

            self.pos_h = pos_h
            self.pos_w = pos_h

            num_patches = pos_h * pos_h

            print("Input resolution {}, overlap resolution {}".format(
                image_size, pos_h))

            self.pos_embedding = nn.Parameter(
                torch.zeros(1, 2, num_patches + 1, dim))
            self.USEPE = True
        else:
            self.USEPE = False

        print("use ape {}".format(self.USEPE))

        self.emb_dim = dim

        self.to_patch_embedding = PatchEmbed3d(kernel_size=patch_size, in_channels=in_channels, embed_dims=dim,
                                               stride=None, padding='corner', input_size=None,
                                               norm_layer=pos_norm)

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))

        self.space_transformer = STransformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=dim * scale_dim, sr_ratio=sr_ratio,
            dropout=dropout)
 

        self.temporal_transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                                mlp_dim=dim * scale_dim, dropout=dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.interpolate_mode = 'bicubic'

        self.diff = AsppFusion(features=dim,dilations=[1,4,12])

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):

        assert patched_img.ndim == 4 and pos_embed.ndim == 4, \
            'the shapes of patched_img and pos_embed must be [B, T,L, C]'
        x_len, pos_len = patched_img.shape[2], pos_embed.shape[2]
        if x_len != pos_len:
            if pos_len == (self.pos_h * self.pos_w) + 1:
                pos_h = self.pos_h
                pos_w = self.pos_w
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return patched_img + pos_embed

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):

        assert pos_embed.ndim == 4, 'shape of pos_embed must be [B,2, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, :, 0]  # [B,2, C]
        pos_embed_weight = pos_embed[:, :, (-1 * pos_h * pos_w):]  # [B,2, L-1, C]

        pos_embed_weight1 = pos_embed_weight[:, 0].reshape(
            1, pos_h, pos_w, pos_embed.shape[3]).permute(0, 3, 1, 2)
        pos_embed_weight1 = resize(
            pos_embed_weight1, size=input_shpae, align_corners=False, mode=mode)

        pos_embed_weight2 = pos_embed_weight[:, 1].reshape(
            1, pos_h, pos_w, pos_embed.shape[3]).permute(0, 3, 1, 2)
        pos_embed_weight2 = resize(
            pos_embed_weight2, size=input_shpae, align_corners=False, mode=mode)

        pos_embed_weight = torch.stack([pos_embed_weight1, pos_embed_weight2], dim=1)

        cls_token_weight = cls_token_weight.unsqueeze(2)
        pos_embed_weight = torch.flatten(pos_embed_weight, 3).transpose(2, 3)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=2)
        return pos_embed
 
    def forward(self, x1, x2):

        x, hw_shape = self.to_patch_embedding(x1, x2)  # b c h w x2 -> b 2 (h w )c

        b, t, n, l = x.shape

        # -------- tokens
        cls_space_tokens = einops.repeat(
            self.space_token, '() n d -> b t n d', b=b, t=t)

        x = torch.cat((cls_space_tokens, x), dim=2)
        # -------- tokens

        if self.USEPE:
            x = self._pos_embeding(x, hw_shape, self.pos_embedding)
            # x += self.pos_embedding[:, :, :(n + 1)]

        x = self.dropout(x)

        x = rearrange(x, 'b t n l -> (b t) n l')

        x = self.space_transformer(x, hw_shape[0], hw_shape[1])  # (b t) n+1 l

        identity = rearrange(x[:, 1:], '(b t) (h w) l -> b t l h w ', b=b, t=t, h=hw_shape[0], w=hw_shape[1], l=l)

        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)  # b t c

        x = self.temporal_transformer(x)  # b t c 此可以操作 decoder 改進

        x = einops.repeat(x, 'b t c -> b t c h w ', h=1, w=1)

        x = identity + identity * x

        # x = rearrange(x, 'b t (h w) l -> b t l h w ', b=b, t=t, h=hw_shape[0], w=hw_shape[1], l=l)  # b t c

        # x1 = x[:, 0]
        # x2 = x[:, 1] 
        # return torch.abs(x1-x2)
        x = einops.repeat(x, 'b t c h w -> b c t  h w ')
        return self.diff(x)
 
if __name__ == '__main__':
    model = VVITv3(image_size=24, dim=96, in_channels=3)
    import torch

    x = torch.randn(1, 3, 24, 24)

    x1  = model(x, x)

    print(x1.shape )
