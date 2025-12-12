import math

from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupScheduler(LRScheduler):
  def __init__(self, optimizer, warmup_steps, last_epoch=-1):
    self.warmup_steps = warmup_steps
    super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    current_step = self.last_epoch + 1
    if current_step <= self.warmup_steps:
      return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]
    return [base_lr for base_lr in self.base_lrs]


class CosineAnnealingLRWLinearWarmupScheduler(LRScheduler):
  def __init__(self, optimizer, warmup_lr, warmup_steps, cosine_annealing_steps, last_epoch=-1):
    self.warmup_lr = warmup_lr
    self.warmup_steps = warmup_steps
    self.cosine_annealing_steps = cosine_annealing_steps
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    """
    Calculate and return the learning rate for each parameter group.
    """
    return [
        (
            (
                self.warmup_lr
                + (base_lr - self.warmup_lr)
                * self.last_epoch
                / self.warmup_steps
            ) if self.last_epoch < self.warmup_steps else
            (
                0.5 * base_lr * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_steps)
                        / (self.cosine_annealing_steps - self.warmup_steps)
                    )
                )
            )
        )
        for base_lr in self.base_lrs
    ]
