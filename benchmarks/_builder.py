from mmcv.utils import Registry

# ----------------------

BACKBONES = Registry('backbone')
SEGHEADS = Registry('seghead')
CHANGES = Registry('change')

BENCHMARKS = Registry('benchmarks')


def build_benchmarks(cfg):
    """Build head."""
    return BENCHMARKS.build(cfg)


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_seghead(cfg):
    return SEGHEADS.build(cfg)


def build_change(cfg):
    return CHANGES.build(cfg)

# ------------------
