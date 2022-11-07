import math
import random

import einops
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import (
    AutoAugmentPolicy,
    InterpolationMode,
    List,
    Optional,
    Tensor,
)


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.normalize(x, x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6, inplace=False)
        return x


def _apply_op(
    img: Tensor,
    op_name: str,
    magnitude: float,
    interpolation: InterpolationMode,
    fill: Optional[List[float]],
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img


def relaxed_bernoulli(logits, temp=0.05, device="cpu"):
    u = torch.rand_like(logits, device=device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits) / temp).sigmoid()


class FreezeSTN(nn.Module):
    """仿射变换"""

    def __init__(self):
        super().__init__()
        self.H = Parameter(torch.randn(6))
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
        grid = torch.nn.functional.affine_grid(H, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        return x


class Alignment:
    def __init__(self, policy_name, img_size, save_path, STN, dataset_type, epoch=10):
        self.policy_name = policy_name
        self.img_size = img_size
        self.stn = STN().cuda()
        self.epoch = epoch
        self.optimizer = torch.optim.AdamW(self.stn.parameters(), 1e-3)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)

        self.criticion = nn.MSELoss()
        self.save_path = save_path

        if dataset_type == "CIFAR":
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        self.tran = transforms.Compose([transforms.Normalize(self.mean, std=self.std)])

    def _augmentation_space(self, num_bins: int, image_size):
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

    def fit(self, dataloader):

        augmentation_space = self._augmentation_space(10, self.img_size)
        self.stn.train()
        for i in range(self.epoch):
            for j, (image, _) in enumerate(dataloader):
                image = image.cuda()
                magnitude = random.random()
                magnitudes, signed = augmentation_space[self.policy_name]
                magnitude_id = min(max(int(magnitude * 10), 0), 9)
                if magnitudes.numel() > 0:
                    magnitude_new = float(magnitudes[magnitude_id].item())
                else:
                    magnitude_new = 0.0
                sign = torch.randint(2, (1,))
                if signed and sign:
                    magnitude_new *= -1.0
                    magnitude *= -1.0
                _image = image.mul(torch.Tensor(self.std)[None, :, None, None].cuda()).add(
                    torch.Tensor(self.mean)[None, :, None, None].cuda()
                )
                _image = _image * 255
                _image = torch.floor(_image + 0.5)
                torch.clip_(_image, 0, 255)
                _image = _image.type(torch.uint8)
                freeze_image = _apply_op(
                    _image,
                    self.policy_name,
                    magnitude_new,
                    interpolation=InterpolationMode.NEAREST,
                    fill=None,
                )
                freeze_image = self.tran(freeze_image / 255).float()
                stn_image = self.stn(image, magnitude, False)
                loss = self.criticion(stn_image, freeze_image)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(f"epoch = {i}, iter = {j}, loss = {round(loss.item(), 3)}")
            torch.save(self.stn.state_dict(), self.save_path)


"""
fangshe: ShearX,ShearY,TranslateX,TranslateY,Rotate,
color: AutoContrast,Invert,Equalize,Solarize,Posterize,Contrast,Color,Brightness,Sharpness,Sample pairing
other: Cutout
"""
