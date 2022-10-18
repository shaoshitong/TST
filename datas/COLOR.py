import random

import einops
import torch.nn as nn
from torchvision import transforms
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


import math

import torch
import torch.nn as nn


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


class ColorAugmentation(nn.Module):
    def __init__(self, ndim=10, scale=1, dataset_type=""):
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
        if dataset_type in ["Equalize", "Invert"]:
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(3), nn.SiLU(), nn.Conv2d(3, 3, (5, 5), (1, 1), (2, 2), bias=False)
            )
        else:
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

    def forward(self, x, magnitude):
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
        if self.conv.requires_grad == False:
            scale, shift = self.sampling(scale, shift)
        return self.conv(self.transform(x, scale, shift))

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


class Alignment:
    def __init__(self, policy_name, img_size, save_path, COLOR, dataset_type,epoch=20):
        self.policy_name = policy_name
        self.img_size = img_size
        if dataset_type == "CIFAR":
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        self.color = COLOR(dataset_type=dataset_type).cuda()
        self.epoch = epoch
        self.optimizer = torch.optim.AdamW(self.color.parameters(), 1e-3, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.999)
        self.criticion = nn.MSELoss()
        self.save_path = save_path
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
        self.color.train()
        for i in range(self.epoch):
            for j, (image, _) in enumerate(dataloader):
                image = image.cuda()
                magnitude = random.random()
                magnitudes, signed = augmentation_space[self.policy_name]
                magnitude_id = min(max(int(magnitude * 10), 0), 9)
                if magnitudes.numel() > 1:
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

                magnitude = (
                    torch.Tensor([magnitude]).to(image.device)[None, ...].expand(image.shape[0], -1)
                )
                stn_image = self.color(image, magnitude)
                loss = self.criticion(stn_image, freeze_image)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(f"epoch = {i}, iter = {j}, loss = {round(loss.item(), 3)}")
            torch.save(self.color.state_dict(), self.save_path)


"""
fangshe: ShearX,ShearY,TranslateX,TranslateY,Rotate,
color: AutoContrast,Invert,Equalize,Solarize,Posterize,Contrast,Color,Brightness,Sharpness,Sample pairing
other: Cutout
"""
