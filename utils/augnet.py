import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datas.LearningAutoAugment import LearningAutoAugment
from torchvision.transforms.autoaugment import AutoAugmentPolicy

class Interploate(nn.Module):
    def __init__(self, img_size):
        super(Interploate, self).__init__()
        self.img_size = img_size

    def forward(self, x):
        return F.interpolate(x, self.img_size, mode='nearest')


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
    def __init__(self, img_size=224):
        super(SmallImageAugNet, self).__init__()

        self.learningautoaugment=LearningAutoAugment(policy=AutoAugmentPolicy.CIFAR10,C=3,H=32,W=32,alpha=0.3)
        self.style_loss=torch.Tensor([0.]).cuda()
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3, img_size * 2 + 1, img_size * 2 + 1))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, img_size * 2 + 1, img_size * 2 + 1))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, img_size * 2 + 3, img_size * 2 + 3))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, img_size * 2 + 3, img_size * 2 + 3))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, img_size * 2 + 5, img_size * 2 + 5))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, img_size * 2 + 5, img_size * 2 + 5))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.norm = nn.InstanceNorm2d(3)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv2d(3, 3, 3, 2).cuda()
        self.spatial = nn.Sequential(nn.ReflectionPad2d((1, 1, 1, 1)),
                                     nn.Conv2d(3, 3, 3, 1, 0),
                                     nn.BatchNorm2d(3),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(3, 3, 3, 2)
                                     ).cuda()
        self.spatial_up = nn.Sequential(Interploate(tuple([2 * img_size + 1, 2 * img_size + 1]))).cuda()

        self.spatial2 = nn.Sequential(nn.ReflectionPad2d((2, 2, 2, 2)),
                                      nn.Conv2d(3, 3, 5, 1, 0),
                                      nn.BatchNorm2d(3),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(3, 3, 5, 2)
                                      ).cuda()
        self.spatial_up2 = nn.Sequential(Interploate(tuple([2 * img_size + 3, 2 * img_size + 3]))).cuda()

        self.spatial3 = nn.Sequential(nn.ReflectionPad2d((3, 3, 3, 3)),
                                      nn.Conv2d(3, 3, 7, 1, 0),
                                      nn.BatchNorm2d(3),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(3, 3, 7, 2)
                                      ).cuda()
        self.spatial_up3 = nn.Sequential(Interploate(tuple([2 * img_size + 5, 2 * img_size + 5]))).cuda()
        self.color = nn.Conv2d(3, 3, 1).cuda()
        for param in list(
                list(self.color.parameters())
                + list(self.spatial.parameters())
                + list(self.spatial_up.parameters())
                + list(self.spatial2.parameters())
                + list(self.spatial_up2.parameters())
                + list(self.spatial3.parameters())
                + list(self.spatial_up3.parameters())
        ):
            param.requires_grad = True

    def gram_matrix(self, input, target):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL
        G_A = torch.matmul(features, features.permute(0, 2, 1)) / (b * c * d)  # a,b,b

        a, b, c, d = target.size()  # a=batch size(=1)
        features = target.view(a, b, c * d)  # resise F_XL into \hat F_XL
        G_B = torch.matmul(features, features.permute(0, 2, 1)) / (c * d)  # a,b,b
        return F.mse_loss(G_B, G_A, reduction='mean')

    def forward(self, x, estimation=False):
        return self.learningautoaugment(x)

        if not estimation:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial_up(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial(x_sdown))
            #
            x_s2down = self.spatial_up2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial2(x_s2down))

            x_s3down = self.spatial_up3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial3(x_s3down))

            weight = torch.rand(4)
            output = (x_c * weight[0] + x_s * weight[1] + x_s2 * weight[2] + x_s3 * weight[3]) / weight.sum()
            return output
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial_up(x )
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial(x_sdown))
            #
            x_s2down = self.spatial_up2(x )
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial2(x_s2down))

            x_s3down = self.spatial_up3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial3(x_s3down))
            output = (x_c + x_s + x_s2 + x_s3) / 4
            x_out_list = [x_c, x_s, x_s2, x_s3]
            self.style_loss = -torch.log(self.gram_matrix(x_out_list[0], x_out_list[1]))\
                              -torch.log(self.gram_matrix(x_out_list[1], x_out_list[2]))\
                              -torch.log(self.gram_matrix(x_out_list[2], x_out_list[3]))
            return output
