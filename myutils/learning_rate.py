import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

    def update_lr(self, args):
        pass


def adjust_learning_rate_StepLR(args, optimizer, current_epoch):
    if current_epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


def adjust_learning_rate_Poly(args, base_lr, optimizer, current_iter, max_iter, power=0.9):
    factor = (1 - current_iter / max_iter) ** power
    args.lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * factor