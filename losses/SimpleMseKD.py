import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from timm.models.layers import DropPath
from torch.autograd import Variable


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


class conv_bn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.FloatTensor([0]).bernoulli_(1 - p_drop).to(x.device)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.FloatTensor(x.size(0)).uniform_(*alpha_range).to(x.device)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.FloatTensor(grad_output.size(0)).uniform_(0, 1).to(grad_output.device)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class Bottleneck(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, p_shakedrop=1.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.shake_drop(out)
        shortcut = x
        featuremap_size = out.size()[2:4]
        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.FloatTensor(
                    batch_size,
                    residual_channel - shortcut_channel,
                    featuremap_size[0],
                    featuremap_size[1],
                ).fill_(0)
            ).to(x.device)
            out = out + torch.cat((shortcut, padding), 1)
        else:
            out = out + shortcut

        return out


class SMSEKD(nn.Module):
    def __init__(self, shapes, out_shapes, in_channels, out_channels, num_classes):
        super().__init__()
        self.in_shapes = shapes  # student
        self.out_shapes = out_shapes  # teacher
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.teacher_embeddings = nn.ModuleList([])
        self.student_embeddings = nn.ModuleList([])
        self.teacher_bns = nn.ModuleList([])
        self.student_bns = nn.ModuleList([])
        self.teacher_fcs = nn.ModuleList([])
        self.student_fcs = nn.ModuleList([])
        for fin, fout in zip(self.in_channels[1:], self.out_channels[1:]):
            self.teacher_embeddings.append(Bottleneck(fout, fout, 1, 1.0))
            self.student_embeddings.append(Bottleneck(fin, fout, 1, 1.0))
            self.teacher_bns.append(
                nn.Sequential(
                    nn.BatchNorm2d(fout),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
            )
            self.student_bns.append(
                nn.Sequential(
                    nn.BatchNorm2d(fout),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
            )
            self.teacher_fcs.append(nn.Linear(fout, num_classes))
            self.student_fcs.append(nn.Linear(fout, num_classes))

    def forward(self, features_student, features_teacher, labels, only_alignment=False):
        features_teacher = features_teacher[1:]
        features_student = features_student[1:]
        l = len(features_teacher)
        for i in range(l):
            features_teacher[i] = self.teacher_embeddings[i](features_teacher[i])
            features_student[i] = self.student_embeddings[i](features_student[i])
            features_teacher[i] = self.teacher_bns[i](features_teacher[i])
            features_student[i] = self.student_bns[i](features_student[i])
            features_teacher[i] = self.teacher_fcs[i](features_teacher[i])
            features_student[i] = self.student_fcs[i](features_student[i])
        kl_loss = lambda pred, target, tem=4: (tem ** 2) * F.kl_div(
            torch.log_softmax(pred / tem, 1), torch.softmax(target / tem, 1), reduction="batchmean"
        )
        total_loss = 0.0
        for i in range(l):
            if not only_alignment:
                m = features_teacher[i].detach()
            else:
                m = features_teacher[i]
            total_loss += kl_loss(features_student[i], m)
        if not only_alignment:
            for i in range(l):
                total_loss += F.cross_entropy(features_teacher[i], labels, reduction="mean")
        return total_loss
