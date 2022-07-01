import monai
from torch import nn


class TEST_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x