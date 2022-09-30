import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            tmpfs = tmpfs.mean(1).flatten(1)
            tmpft = tmpft.mean(1).flatten(1)

            loss += inter_class_relation(tmpfs, tmpft) * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


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
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1, stride=1),
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
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x

    #
    # def mix_student_and_teacher(
    #         self, feature_map1, feature_map2, soft_mask,
    # ) -> torch.Tensor:
    #     """
    #     Here, we perform a completely random mask
    #     """
    #     b, c, h, w = feature_map1.shape
    #     patch_size = 7 if feature_map1.shape[-1] % 7 == 0 else 4
    #     soft_mask = soft_mask.expand(-1, -1, patch_size, patch_size, -1, -1)
    #     soft_mask = rearrange(soft_mask, "b c p q h w -> b c (p h) (q w)")
    #     hard_mask = soft_mask > torch.rand_like(soft_mask).to(soft_mask.device)
    #     new_feature_map = soft_mask * feature_map1 + (1 - soft_mask) * feature_map2
    #     return torch.where(hard_mask, feature_map1, feature_map2).detach() + new_feature_map - new_feature_map.detach()


class ReviewKD(nn.Module):
    def __init__(self, shapes, out_shapes, in_channels, out_channels, max_mid_channel):
        super().__init__()
        self.shapes = shapes[::-1]
        self.out_shapes = out_shapes[::-1]
        in_channels = in_channels
        out_channels = out_channels
        self.max_mid_channel = max_mid_channel

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]

    def forward(self, features_student, features_teacher):
        # get features
        x = features_student[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        # losses
        loss_reviewkd = hcl_loss(results, features_teacher)

        return loss_reviewkd
