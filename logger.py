from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class CustomLogger:
    def __init__(self, log_dir=None, comment=None):
        self.logger = SummaryWriter(log_dir=log_dir, comment=comment)
        self.stats = defaultdict(list)

    def log(self, **kwargs):
        for key, val in kwargs.items():
            self.stats[key].append(val)
            self.logger.add_scalar(f'stats/{key}', val, len(self.stats[key]) - 1)


