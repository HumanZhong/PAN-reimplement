import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.current_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.current_val = val
        self.sum += self.current_val * n
        self.count += n
        self.avg = self.sum / self.count