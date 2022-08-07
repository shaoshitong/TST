import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

from datas.LearningAutoAugment import LearningAutoAugment


class Interploate(nn.Module):
    def __init__(self, img_size):
        super(Interploate, self).__init__()
        self.img_size = img_size

    def forward(self, x):
        return F.interpolate(x, self.img_size, mode="nearest")


class BigImageAugNet(nn.Module):
    def __init__(self, img_size=224):
        super(BigImageAugNet, self).__init__()
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3, img_size - 8, img_size - 8))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, img_size - 8, img_size - 8))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, img_size - 12, img_size - 12))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, img_size - 12, img_size - 12))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, img_size - 16, img_size - 16))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, img_size - 16, img_size - 16))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(3, img_size - 4, img_size - 4))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, img_size - 4, img_size - 4))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        self.norm = nn.InstanceNorm2d(3)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv2d(3, 3, 9).cuda()
        self.spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

        self.spatial2 = nn.Conv2d(3, 3, 13).cuda()
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

        self.spatial3 = nn.Conv2d(3, 3, 17).cuda()
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()

        self.spatial4 = nn.Conv2d(3, 3, 5).cuda()
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()
        self.color = nn.Conv2d(3, 3, 1).cuda()

        for param in list(
            list(self.color.parameters())
            + list(self.spatial.parameters())
            + list(self.spatial_up.parameters())
            + list(self.spatial2.parameters())
            + list(self.spatial_up2.parameters())
            + list(self.spatial3.parameters())
            + list(self.spatial_up3.parameters())
            + list(self.spatial4.parameters())
            + list(self.spatial_up4.parameters())
        ):
            param.requires_grad = False

    def forward(self, x, estimation=False):
        if not estimation:
            spatial = nn.Conv2d(3, 3, 9).cuda()
            spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

            spatial2 = nn.Conv2d(3, 3, 13).cuda()
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

            spatial3 = nn.Conv2d(3, 3, 17).cuda()
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()

            spatial4 = nn.Conv2d(3, 3, 5).cuda()
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()

            color = nn.Conv2d(3, 3, 1).cuda()
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(F.dropout(color(x), p=0.2))

            x_sdown = spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(spatial_up(x_sdown))
            #
            x_s2down = spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(spatial_up2(x_s2down))
            #
            #
            x_s3down = spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(spatial_up3(x_s3down))

            #
            x_s4down = spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(spatial_up4(x_s4down))

            output = (
                weight[0] * x_c
                + weight[1] * x_s
                + weight[2] * x_s2
                + weight[3] * x_s3
                + weight[4] * x_s4
            ) / weight.sum()
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial_up(x_sdown))
            #
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(self.spatial_up4(x_s4down))

            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output


class SmallImageAugNet(nn.Module):
    def __init__(self, img_size=224, num_train_samples=50000, yaml=None):
        super(SmallImageAugNet, self).__init__()
        # TODO: M
        self.alpha = 1
        self.learningautoaugment = LearningAutoAugment(
            policy=AutoAugmentPolicy.CIFAR10,
            C=yaml["LAA"]["C"],
            H=yaml["LAA"]["H"],
            W=yaml["LAA"]["W"],
            p=yaml["LAA"]["p"],
            num_train_samples=num_train_samples,
        )
        print(f"alpha is {self.alpha}")
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3, img_size, img_size))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, img_size, img_size))
        nn.init.normal_(self.shift_mean, 0, 0.1)
        self.norm = nn.InstanceNorm2d(3)
        self.color = nn.Conv2d(3, 3, 1).cuda()
        self.tran = (
            transforms.Compose(
                [transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            )
            if yaml["augmentation_policy"] == "cifar10"
            else transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    def forward(self, x,y, indexs, epoch):
        x,y = self.learningautoaugment(x,y, indexs, epoch)
        return x,y
