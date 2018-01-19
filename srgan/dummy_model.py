import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, x):
        return torch.autograd.Variable(torch.Tensor(1))
