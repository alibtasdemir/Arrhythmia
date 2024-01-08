import torch.nn as nn
import numpy as np
import torch


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Sigmoid(nn.Module):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class ReLU(nn.Module):
    def forward(self, x):
        return np.maximum(0, x)


class LeakyReLU(nn.Module):
    def forward(self, x, alpha=0.1):
        return np.maximum(alpha*x, x)


class Tanh(nn.Module):
    def forward(self, x):
        return np.tanh(x)


