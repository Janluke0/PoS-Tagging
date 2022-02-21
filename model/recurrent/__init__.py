from torch import nn


class OneHot(nn.Module):

    def __init__(self, nclass) -> None:
        super(type(self), self).__init__()
        self.nclass = nclass

    def forward(self, x):
        return nn.functional.one_hot(x, num_classes=self.nclass).float()
