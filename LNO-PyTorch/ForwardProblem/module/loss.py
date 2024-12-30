import torch


class RelLpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(RelLpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.sum(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        target = torch.sum(abs(target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        rloss = torch.mean(error / target)
        return rloss
    

class LpLoss(torch.nn.modules.loss._Loss):
    def __init__(self, p):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = torch.mean(abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        loss = torch.mean(error)
        return loss
