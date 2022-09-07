# # ---------------------------------------------------------------------------------------------
# # Modified from Swin Transformer, please refer to https://github.com/microsoft/Swin-Transformer
# # ---------------------------------------------------------------------------------------------
#
# import torch
# from timm.scheduler.cosine_lr import CosineLRScheduler
# from timm.scheduler.step_lr import StepLRScheduler
# from timm.scheduler.scheduler import Scheduler
#
#
# class LinearLRScheduler(Scheduler):
#     def __init__(self,
#                  optimizer: torch.optim.Optimizer,
#                  t_initial: int,
#                  lr_min_rate: float,
#                  warmup_t=0,
#                  warmup_lr_init=0.,
#                  t_in_epochs=True,
#                  noise_range_t=None,
#                  noise_pct=0.67,
#                  noise_std=1.0,
#                  noise_seed=42,
#                  initialize=True,
#                  ) -> None:
#         super().__init__(
#             optimizer, param_group_field="lr",
#             noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
#             initialize=initialize)
#
#         self.t_initial = t_initial
#         self.lr_min_rate = lr_min_rate
#         self.warmup_t = warmup_t
#         self.warmup_lr_init = warmup_lr_init
#         self.t_in_epochs = t_in_epochs
#         if self.warmup_t:
#             self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
#             super().update_groups(self.warmup_lr_init)
#         else:
#             self.warmup_steps = [1 for _ in self.base_values]
#
#     def _get_lr(self, t):
#         if t < self.warmup_t:
#             lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
#         else:
#             t = t - self.warmup_t
#             total_t = self.t_initial - self.warmup_t
#             lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
#         return lrs
#
#     def get_epoch_values(self, epoch: int):
#         if self.t_in_epochs:
#             return self._get_lr(epoch)
#         else:
#             return None
#
#     def get_update_values(self, num_updates: int):
#         if not self.t_in_epochs:
#             return self._get_lr(num_updates)
#         else:
#             return None
#
#
# def build_scheduler(config, optimizer):
#     num_steps = int(config.TRAIN.EPOCHS)
#     warmup_steps = int(config.TRAIN.WARMUP_EPOCHS)
#     decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS)
#
#     lr_scheduler = None
#     if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
#         lr_scheduler = CosineLRScheduler(
#             optimizer,
#             t_initial=num_steps,
#             # t_mul=1., #新版timm没有
#             lr_min=config.TRAIN.MIN_LR,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             cycle_limit=1,
#             t_in_epochs=config.TRAIN.T_IN_EPOCHS,
#         )
#     elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
#         lr_scheduler = LinearLRScheduler(
#             optimizer,
#             t_initial=num_steps,
#             lr_min_rate=0.01,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             t_in_epochs=config.TRAIN.T_IN_EPOCHS,
#         )
#     elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
#         lr_scheduler = StepLRScheduler(
#             optimizer,
#             decay_t=decay_steps,
#             decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             t_in_epochs=config.TRAIN.T_IN_EPOCHS,
#         )
#
#     return lr_scheduler
#
#
# if __name__ == "__main__":
#     # import matplotlib as mpl
#     import torch
#     import matplotlib.pyplot as plt
#
#     lr = 0.001
#     net = torch.nn.Conv2d(1, 1, 1)
#
#     epochs = 250
#
#     plt.figure()
#
#     from omegaconf import OmegaConf
#     from trainers.utils.optimizer import build_optimizer
#
#     config = OmegaConf.create(
#         dict(
#             TRAIN=dict(
#                 EPOCHS=epochs,
#                 WARMUP_EPOCHS=6,
#                 T_IN_EPOCHS=True,
#                 WARMUP_LR=1e-6,
#                 MIN_LR=1e-7,
#                 BASE_LR=lr,
#                 WEIGHT_DECAY=0.002,
#                 OPTIMIZER=dict(
#                     NAME='adamw',
#                     MOMENTUM=0.9,
#                     EPS=1e-8,
#                     BETAS=[0.9, 0.99]
#                 ),
#                 LR_SCHEDULER=dict(
#                     NAME='step',
#                     DECAY_RATE=0.1,
#                     DECAY_EPOCHS=20
#                 )
#             )
#         )
#     )
#     opt = build_optimizer(config, parameters=net.parameters())
#     scheduler = build_scheduler(config, optimizer=opt)
#
#     lrs = []
#     for epoch in range(epochs):
#         lrs.append(scheduler.get_epoch_values(epoch))
#         scheduler.step(epoch)
#     plt.plot(range(epochs), lrs, label=config.TRAIN.LR_SCHEDULER.NAME)
#
#     plt.legend()
#     plt.show()
#
#     # timm 使用方法简单版本 https://github.com/PyTorchLightning/pytorch-lightning/issues/5555
#     # def configure_optimizers(self):
#     #
#     #     return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]
#     #
#     # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
#     #     print(self.current_epoch)
#     #     scheduler.step(
#     #         epoch=self.current_epoch
#     #     )  # timm's scheduler need the epoch va
