import random

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

class stnGenerator(nn.Module):
    ''' 仿射变换 '''

    def __init__(self, zdim=10, imsize=[32, 32], mode='translate'):
        super().__init__()
        self.mode = mode
        self.zdim = zdim

        self.mapz = nn.Linear(zdim, imsize[0] * imsize[1])
        if imsize == [32, 32]:
            self.loc = nn.Sequential(
                nn.Conv2d(4, 16, 5), nn.AvgPool2d(2), nn.ReLU(),
                nn.Conv2d(16, 32, 5), nn.AvgPool2d(2), nn.ReLU(), )
            self.fc_loc = nn.Sequential(
                nn.Linear(32 * 5 * 5, 32), nn.ReLU(),
                nn.Linear(32, 6))
        elif imsize == [224, 224]:
            self.loc = nn.Sequential(
                nn.Conv2d(4, 16, 7, 2), nn.AvgPool2d(2), nn.ReLU(),
                nn.Conv2d(16, 32, 7, 2), nn.AvgPool2d(2), nn.ReLU(), )
            self.fc_loc = nn.Sequential(
                nn.Linear(32 * 12 * 12, 32), nn.ReLU(),
                nn.Linear(32, 6))
        # init the weight
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))

    def forward(self, x, return_H=False):
        z = torch.randn(len(x), self.zdim).to(x.device)
        z = self.mapz(z).view(len(x), 1, x.size(2), x.size(3))
        loc = self.loc(torch.cat([x, z], dim=1))  # [N, -1]
        loc = loc.view(len(loc), -1)
        H = self.fc_loc(loc) * 2
        H = H.view(len(H), 2, 3)
        H[:,0,2] = torch.tanh(H[:,0,2])/5
        H[:,1,2] = torch.tanh(H[:,1,2])/5
        if self.mode == 'translate':
            H[:, 0, 0] = 1
            H[:, 0, 1] = 0
            H[:, 1, 0] = 0
            H[:, 1, 1] = 1
        grid = F.affine_grid(H, x.size())
        x = F.grid_sample(x, grid)
        if return_H:
            return x, H
        else:
            return x


class TranslateNet(nn.Module):
    def __init__(self, type="imagenet"):
        super(TranslateNet, self).__init__()
        ############# Trainable Parameters
        self.trans = (
            transforms.Compose(
                [transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            )
            if type == "cifar10"
            else transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.stngen = stnGenerator(imsize=[224, 224])
        self.iter = 0
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.interpolate(x, size=[224, 224], mode="nearest")
        x = self.stngen(x)
        output = F.interpolate(x, [h, w])
        if random.random()>0.9:
            from utils.save_Image import change_tensor_to_image
            change_tensor_to_image(output[0],"images",f"{self.iter}")
            self.iter+=1
        return output
