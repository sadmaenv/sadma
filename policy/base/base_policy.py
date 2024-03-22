from utils.utils import ValueNorm

class BasePolicy:
    def __init__(self, args, mac):
        self.args = args
        self.mac = mac
        self.device = args.train_device
        self.optimizer = None
        self.use_value_norm = getattr(self.args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.device)

    def lr_decay(self, lr, current_step):
        lr = lr - (lr * (current_step / float(self.args.max_steps)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self):
        pass
    
    def save_model(self):
        pass