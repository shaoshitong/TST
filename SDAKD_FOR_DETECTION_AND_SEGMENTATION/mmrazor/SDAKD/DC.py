import random,math
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



class DetectionColorAugmentation(nn.Module):
    def __init__(self, ndim=10, scale=1):
        super().__init__()

        linear = lambda ic, io: nn.Linear(ic, io, False)
        n_hidden = 1 + 10
        self.n_regress = linear(n_hidden, 2)
        self.c_regress = linear(n_hidden, 2)
        self.scale = nn.Parameter(torch.Tensor([scale]))
        self.relax = True
        self.stochastic = True
        self.logits = nn.Parameter(torch.zeros(1))
        self.feature = nn.Parameter(torch.randn(10))
        self.ndim = ndim
        self.conv = nn.Conv2d(3, 3, (5, 5), (1, 1), (2, 2), bias=False)

    def sampling(self, scale, shift, temp=0.05):
        if self.stochastic:  # random apply
            logits = self.logits.repeat(scale.shape[0]).reshape(-1, 1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=scale.device)
            if not self.relax:  # hard sampling
                prob = (prob > 0.5).float()
            scale = 1 - prob + prob * scale
            shift = prob * shift  # omit "+ (1 - prob) * 0"
        return scale, shift

    def forward(self, x, magnitude, boxes, re=True):
        noise = self.feature + torch.randn_like(self.feature).to(self.feature.data.device) / 100
        if isinstance(magnitude, (float, int)):
            magnitude = torch.Tensor([magnitude]).to(x.device)
            magnitude = magnitude.view(1, -1).expand(x.shape[0], -1)
        noise = noise.view(1, -1).expand(x.shape[0], -1)

        noise = torch.cat([noise, magnitude], 1)
        gfactor = self.n_regress(noise).reshape(-1, 2, 1, 1)
        factor = self.c_regress(noise).reshape(-1, 2, 1, 1)

        scale, shift = factor.chunk(2, dim=1)
        g_scale, g_shift = gfactor.chunk(2, dim=1)
        scale = (g_scale + scale).sigmoid()
        shift = (g_shift + shift).sigmoid()
        # scaling
        scale = self.scale * (scale - 0.5) + 1
        shift = shift - 0.5
        # random apply
        if re == True:
            scale, shift = self.sampling(scale, shift)
        return self.conv(self.transform(x, scale, shift)), boxes

    def transform(self, x, scale, shift):
        # ignore zero padding region
        with torch.no_grad():
            h, w = x.shape[-2:]
            mask = (x.sum(1, keepdim=True) == 0).float()  # mask pixels having (0, 0, 0) color
            mask = torch.logical_and(
                mask.sum(-1, keepdim=True) < w, mask.sum(-2, keepdim=True) < h
            )  # mask zero padding region

        x = (scale * x + shift) * mask
        return x
