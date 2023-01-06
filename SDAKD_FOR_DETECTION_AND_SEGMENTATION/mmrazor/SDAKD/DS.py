import random, math
import torch
import torch.nn as nn
import torch.nn.functional as F


def relaxed_bernoulli(logits, temp=0.05, device="cpu"):
    u = torch.rand_like(logits, device=device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits) / temp).sigmoid()


class TriangleWave(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        o = torch.acos(torch.cos(x * math.pi)) / math.pi
        self.save_for_backward(x)
        return o

    @staticmethod
    def backward(self, grad):
        o = self.saved_tensors[0]
        # avoid nan gradient at the peak by replacing it with the right derivative
        o = torch.floor(o) % 2
        grad[o == 1] *= -1
        return grad


class DetectionFreezeSTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = nn.Parameter(torch.randn(6))
        self.fc = nn.Linear(7, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))
        self.logits = nn.Parameter(torch.zeros(1))
        self.register_buffer("i_matrix", torch.Tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3))

    def sample(self, A, temp=0.05):
        logits = self.logits.repeat(A.shape[0]).reshape(-1, 1, 1)
        prob = relaxed_bernoulli(logits, temp, device=logits.device)
        return (1 - prob) * self.i_matrix + prob * A

    def forward(self, x, magnitude, rg=True):
        if isinstance(magnitude, (float, int)):
            magnitude = torch.Tensor([magnitude]).to(x.device)
            magnitude = magnitude.view(1, -1).expand(x.shape[0], -1)
        H = self.H[None, ...].expand(x.shape[0], -1)
        H = torch.cat([H, magnitude], 1)
        if rg == True:
            H = H + torch.randn_like(H).to(H.device) / 100
        H = self.fc(H).view(-1, 2, 3)
        if rg == True:
            H = self.sample(H)
        return H
