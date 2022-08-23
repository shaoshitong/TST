import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x


class TWReviewKD(nn.Module):
    def __init__(self, shapes, out_shapes, in_channels, out_channels, max_mid_channel):
        super().__init__()
        self.shapes = copy.deepcopy(shapes[::-1])
        self.out_shapes = copy.deepcopy(out_shapes[::-1])
        in_channels = in_channels # student
        out_channels = out_channels # teacher
        self.max_mid_channel = max_mid_channel

        abfs_student = nn.ModuleList()
        abfs_teacher = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs_student.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
            abfs_teacher.append(
                ABF(
                    out_channels[idx],
                    mid_channel,
                    in_channel,
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs_student = abfs_student[::-1]
        self.abfs_teacher = abfs_teacher[::-1]

    def forward(self, features_student, features_teacher):
        # get features
        features_student = features_student[:-1] + [features_student[-1]]
        features_teacher = features_teacher[:-1] + [features_teacher[-1]]
        features_student = features_student[::-1]
        features_teacher = features_teacher[::-1]
        student_results = []
        out_features, res_features = self.abfs_student[0](features_student[0], out_shape=self.out_shapes[0])
        student_results.append(out_features)
        for features, abf, shape, out_shape in zip(
            features_student[1:], self.abfs_student[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            student_results.insert(0, out_features)

        "==================================="

        teacher_results = []
        out_features, res_features = self.abfs_teacher[0](features_teacher[0], out_shape=self.shapes[0])
        teacher_results.append(out_features)
        for features, abf, shape, out_shape in zip(
            features_teacher[1:], self.abfs_teacher[1:], self.out_shapes[1:], self.shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            teacher_results.insert(0, out_features)
        # losses
        losses = 0.
        for s,t in zip(features_student[::-1],teacher_results):
            losses += F.mse_loss(s,t,reduction="mean")
        for s,t in zip(features_teacher[::-1],student_results):
            losses += F.mse_loss(s,t,reduction="mean")
        return losses
