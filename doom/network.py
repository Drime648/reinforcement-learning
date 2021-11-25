import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    """Some Information about Network"""
    def __init__(self, input_shape, num_actions):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d()

    def forward(self, x):

        return x