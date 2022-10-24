# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .changes import *  # noqa: F401,F403
from .builder import (BACKBONES,CHANGES, HEADS, LOSSES, SEGMENTORS, build_backbone,build_change,
                      build_head, build_loss, build_segmentor)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .aux_heads import *
__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor','build_change'
]