import paddle


class RelLpLoss(paddle.nn.Layer):
    def __init__(self, p):
        super(RelLpLoss, self).__init__()
        self.p = p

    def forward(self, pred, target):
        error = paddle.sum(paddle.abs(pred - target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        target = paddle.sum(paddle.abs(target) ** self.p, tuple(range(1, len(pred.shape)))) ** (1/self.p)
        rloss = paddle.mean(error / target)
        return rloss
