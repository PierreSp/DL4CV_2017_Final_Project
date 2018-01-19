import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = torch.nn.Conv2D(1, 1, 1)

    def forward(self, x):
        return torch.autograd.Variable(torch.Tensor(1))
