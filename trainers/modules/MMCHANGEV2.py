from abc import ABC

from .MMCHANGE import MMCHANGE
from trainers.utils.lr_scheduler_torch import build_scheduler
from trainers.utils.optimizer import build_optimizer


class MMCHANGEV2(MMCHANGE, ABC):
    def __init__(self, **kwargs):
        super(MMCHANGEV2, self).__init__(**kwargs)

    def configure_optimizers(self):
        param_groups = self.model.get_param_groups()  # backbone; backbone_norm; change; (decode_head,neck,auxiliary_head)
        params = [
            {
                "params": param_groups[0],
                "lr": self.hparams.TRAIN.BASE_LR,
                "weight_decay": self.hparams.TRAIN.WEIGHT_DECAY,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],  # change;
                "lr": self.hparams.TRAIN.BASE_LR,
                "weight_decay": self.hparams.TRAIN.WEIGHT_DECAY,
            },
            {
                "params": param_groups[3],  # seg_head;change_head;neck
                "lr": self.hparams.TRAIN.BASE_LR * 5,
                "weight_decay": self.hparams.TRAIN.WEIGHT_DECAY,
            },
        ]
        optimizer = build_optimizer(self.hparams, parameters=params)
        scheduler = build_scheduler(self.hparams, optimizer=optimizer)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.TRAIN.INTERVAL,
                "monitor": self.hparams.TRAIN.MONITOR
            },
        }
