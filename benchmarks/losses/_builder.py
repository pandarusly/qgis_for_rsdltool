from mmcv.utils import Registry

LOSSES = Registry('loss')


def build_losses(cfg):
    """Build head."""
    return LOSSES.build(cfg)
