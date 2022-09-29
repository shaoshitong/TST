
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import einops
from einops import rearrange

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


class conv_bn(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,(1,1),(1,1),(0,0),bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self,x):
        x = self.bn(self.conv(x))
        return x

class SMSEKD(nn.Module):
    def __init__(self, shapes, out_shapes, in_channels, out_channels, max_mid_channel):
        super().__init__()
        self.in_shapes = shapes # student
        self.out_shapes = out_shapes # teacher
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_mid_channel = max_mid_channel
        self.embeddings = nn.ModuleList([])
        for fin,fout in zip(self.in_channels,self.out_channels):
            self.embeddings.append(conv_bn(fin,fout))
        for parameter in self.embeddings.parameters():
            parameter.requires_grad = False

    def forward(self, features_student, features_teacher):
        embeddings = []
        for i,feature in enumerate(features_student):
            embeddings.append(self.embeddings[i](feature))
        mse_1 = F.mse_loss(embeddings[0],features_teacher[0])
        mse_3 = F.mse_loss(embeddings[2],features_teacher[2])
        return mse_3+mse_1
