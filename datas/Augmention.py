import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms
from torchvision.transforms.autoaugment import (
    AutoAugmentPolicy,
    InterpolationMode,
    List,
    Optional,
    Tensor,
)

from .COLOR import ColorAugmentation, _apply_op
from .STN import FreezeSTN
from utils.save_Image import change_tensor_to_image

def relaxed_bernoulli(logits, temp=0.05):
    u = torch.rand_like(logits, device=logits.device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits) / temp).sigmoid()


def pre_tran(image, mean, std):
    _image = image.mul(torch.Tensor(std)[None, :, None, None].cuda()).add(
        torch.Tensor(mean)[None, :, None, None].cuda()
    )
    _image = _image * 255
    _image = torch.floor(_image + 0.5)
    torch.clip_(_image, 0, 255)
    _image = _image.type(torch.uint8)
    return _image


class LAMBDA_AUG(nn.Module):
    def __init__(self, dataset_type, lambda_function):
        super(LAMBDA_AUG, self).__init__()
        if dataset_type == "CIFAR":
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        self.tran = transforms.Normalize(mean=self.mean, std=self.std)
        self.pre_tran = lambda x: pre_tran(x, self.mean, self.std)
        self.after_tran = lambda x: self.tran(x / 255)
        self.aug = lambda_function

    def forward(self, x):
        x = self.pre_tran(x)
        x = self.aug(x)
        x = self.after_tran(x)
        return x


def _gen_cutout_coord(height, width, size):
    height_loc = random.randint(0, height - 1)
    width_loc = random.randint(0, width - 1)

    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(height, height_loc + size // 2), min(width, width_loc + size // 2))

    return upper_coord, lower_coord


class Cutout(torch.nn.Module):
    def __init__(self, size=16):
        super().__init__()
        self.size = size

    def forward(self, img):
        h, w = img.shape[-2:]
        upper_coord, lower_coord = _gen_cutout_coord(h, w, self.size)

        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = torch.ones_like(img)
        mask[..., upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1]] = 0
        return img * mask


class Mulit_Augmentation(nn.Module):
    # TODO: LEARNING SUB POLICIES
    LEARNING_COLOR_LIST = ["Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize"]
    NO_LEARNING_COLOR_LIST = ["Equalize", "Invert"]
    LEARNING_STN_LIST = ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]
    OTHER_LIST = ["CUTMIX"]

    def __init__(self, pretrain_path, dataset_type, solve_number):
        super(Mulit_Augmentation, self).__init__()
        self.len_policies = len(self.LEARNING_STN_LIST) + len(self.LEARNING_COLOR_LIST)
        self.probabilities = Parameter(torch.zeros(
                self.len_policies + len(self.NO_LEARNING_COLOR_LIST) + len(self.OTHER_LIST))
        )
        self.magnitudes = Parameter(torch.zeros(self.len_policies))
        self.pretrain_path = pretrain_path
        self.dataset_type = dataset_type
        self.solve_number = solve_number
        concat_path = lambda x, path=self.pretrain_path: os.path.join(path, x) + ".pth"

        self.learning_color_model_list = nn.ModuleList([])
        self.learning_stn_model_list = nn.ModuleList([])
        self.nolearning_model_list = []

        for name in self.LEARNING_STN_LIST:
            path = concat_path(name)
            state_dict = torch.load(path)
            model = FreezeSTN()
            model.load_state_dict(state_dict)
            self.learning_stn_model_list.append(model)

        for name in self.LEARNING_COLOR_LIST:
            path = concat_path(name)
            state_dict = torch.load(path)
            model = ColorAugmentation(dataset_type=dataset_type)
            model.load_state_dict(state_dict)
            self.learning_color_model_list.append(model)

        self._freeze_parameter()  # TODO: FREEZE

        equailize = LAMBDA_AUG(
            dataset_type=dataset_type,
            lambda_function=lambda _image: _apply_op(
                _image, "Equalize", 0.0, interpolation=InterpolationMode.NEAREST, fill=None
            ),
        )
        self.nolearning_model_list.append(equailize)
        invert = LAMBDA_AUG(
            dataset_type=dataset_type,
            lambda_function=lambda _image: _apply_op(
                _image, "Invert", 0.0, interpolation=InterpolationMode.NEAREST, fill=None
            ),
        )
        self.nolearning_model_list.append(invert)
        _cutout = Cutout(32 if dataset_type == "CIFAR" else 224)
        self.nolearning_model_list.append(_cutout)

    def _freeze_parameter(self):
        for parameter in self.learning_stn_model_list.parameters():
            parameter.requires_grad = False
        for parameter in self.learning_color_model_list.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _clamp(self):
        EPS = 1e-8
        self.probabilities.data = torch.clamp(self.probabilities.data, EPS, 1 - EPS)
        self.magnitudes.data = torch.clamp(self.magnitudes.data, EPS, 1 - EPS)

    def forward(self, image):
        p = torch.sigmoid(self.probabilities)
        m = torch.sigmoid(self.magnitudes)
        p = relaxed_bernoulli(p)
        len = p.shape[0]
        index = torch.randperm(len).to(image.device)
        index = index[: self.solve_number].tolist()
        result = []
        p_iter = 0
        m_iter = 0
        for tran in self.learning_color_model_list:
            if p_iter in index:
                _m = m[p_iter].view(-1, 1).expand(image.shape[0], -1)
                now_image = tran(image, _m)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                result.append(now_image)
                change_tensor_to_image(now_image[0],'image',f'{p_iter}')
            p_iter += 1
            m_iter += 1

        for tran in self.learning_stn_model_list:
            if p_iter in index:
                _m = m[p_iter].view(-1, 1).expand(image.shape[0], -1)
                now_image = tran(image, _m)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                result.append(now_image)
                change_tensor_to_image(now_image[0],'image',f'{p_iter}')
            p_iter += 1
            m_iter += 1

        for tran in self.nolearning_model_list:
            if p_iter in index:
                now_image = tran(image)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                result.append(now_image)
                change_tensor_to_image(now_image[0],'image',f'{p_iter}')
            p_iter += 1

        result = torch.stack(result).mean(0)
        return result
