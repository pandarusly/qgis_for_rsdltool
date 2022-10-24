import torch.nn as nn
from mmcv.runner import BaseModule
import torch
from ..builder import CHANGES, build_plugin
from ...ops import resize


@CHANGES.register_module()
class AdvanceChange(BaseModule):

    def __init__(self,
                 in_channels,
                 in_index=-1,
                 input_transform=None,
                 align_corners=False,
                 fusion_forms=("concate", "concate", "concate", "concate"),
                 **kwargs
                 ):
        super(AdvanceChange, self).__init__(init_cfg=dict(type="Normal", std=0.01))
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        if isinstance(fusion_forms, dict):
            self.fusion = nn.ModuleList()
            self.fusion.append(build_plugin(fusion_forms))
        elif isinstance(fusion_forms, (list, tuple)):
            self.fusion = nn.ModuleList()
            for fusion in fusion_forms:
                self.fusion.append(build_plugin(fusion))
        else:
            raise TypeError(
                f"loss_decode must be a dict or sequence of dict,\
                but got {type(fusion_forms)}"
            )

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ["resize_concat", "multiple_select"]
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):

        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, x1, x2):
        x1, x2 = self._transform_inputs(x1), self._transform_inputs(x2)
        out = []
        for i, fuse in enumerate(self.fusion):
            out.append(fuse(x1[i], x2[i]))
        return out
