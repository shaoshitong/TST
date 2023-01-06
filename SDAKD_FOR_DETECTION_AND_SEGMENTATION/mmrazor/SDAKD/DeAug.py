import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmdet.datasets.pipelines.transforms import CutOut
import mmcv
from .DC import DetectionColorAugmentation
from .DS import DetectionFreezeSTN
from .AP import _apply_op


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
    def __init__(self, lambda_function):
        """
        Args:
            lambda_function: a callable function
        Dataset:
            only applied for MS-COCO
        """
        super(LAMBDA_AUG, self).__init__()
        self.mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        self.std = [58.395 / 255, 57.12 / 255, 57.375 / 255]
        self.tran = transforms.Normalize(mean=self.mean, std=self.std)
        self.pre_tran = lambda x: pre_tran(x, self.mean, self.std)
        self.after_tran = lambda x: self.tran(x / 255)
        self.aug = lambda_function

    def forward(self, x, boxes):
        x = self.pre_tran(x)
        x = self.aug(x)
        x = self.after_tran(x)
        return x, boxes


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

    def forward(self, img, boxes):
        h, w = img.shape[-2:]
        upper_coord, lower_coord = _gen_cutout_coord(h, w, self.size)

        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = torch.ones_like(img)
        mask[..., upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1]] = 0
        return img * mask, boxes


class Mulit_Augmentation(nn.Module):
    # TODO: LEARNING SUB POLICIES
    LEARNING_COLOR_LIST = ["Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize"]
    NO_LEARNING_COLOR_LIST = ["Equalize", "Invert"]
    LEARNING_STN_LIST = ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]
    OTHER_LIST = ["CUTMIX"]

    def __init__(self, pretrain_path, solve_number):
        super(Mulit_Augmentation, self).__init__()
        self.len_policies = len(self.LEARNING_STN_LIST) + len(self.LEARNING_COLOR_LIST)
        self.probabilities = Parameter(
            torch.zeros(self.len_policies + len(self.NO_LEARNING_COLOR_LIST) + len(self.OTHER_LIST))
        )
        self.magnitudes = Parameter(torch.zeros(self.len_policies))
        self.pretrain_path = pretrain_path
        self.solve_number = solve_number
        concat_path = lambda x, path=self.pretrain_path: os.path.join(path, x) + ".pth"

        self.learning_color_model_list = nn.ModuleList([])
        self.learning_stn_model_list = nn.ModuleList([])
        self.nolearning_model_list = []

        for name in self.LEARNING_STN_LIST:
            path = concat_path(name)
            state_dict = torch.load(path, map_location="cpu")
            model = DetectionFreezeSTN()
            model.load_state_dict(state_dict)
            self.learning_stn_model_list.append(model)

        for name in self.LEARNING_COLOR_LIST:
            path = concat_path(name)
            state_dict = torch.load(path, map_location="cpu")
            model = DetectionColorAugmentation()
            model.load_state_dict(state_dict)
            self.learning_color_model_list.append(model)

        self._freeze_parameter()  # TODO: FREEZE

        equailize = LAMBDA_AUG(
            lambda_function=lambda _image: _apply_op(
                _image, "Equalize", 0.0, interpolation=InterpolationMode.NEAREST, fill=None
            ),
        )
        self.nolearning_model_list.append(equailize)
        invert = LAMBDA_AUG(
            lambda_function=lambda _image: _apply_op(
                _image, "Invert", 0.0, interpolation=InterpolationMode.NEAREST, fill=None
            ),
        )
        self.nolearning_model_list.append(invert)
        _cutout = Cutout(128)
        self.nolearning_model_list.append(_cutout)
        self.iteration = 0

    def _freeze_parameter(self):

        for parameter in self.learning_stn_model_list.parameters():
            parameter.requires_grad = False
        for parameter in self.learning_color_model_list.parameters():
            parameter.requires_grad = False

    def print_magnitudes(self):
        magnitudes = torch.sigmoid(self.magnitudes.data)
        keys = self.LEARNING_COLOR_LIST + self.LEARNING_STN_LIST
        result_str = ""
        for key, magnitude in zip(keys, magnitudes):
            result_str += f"key is {key}, magnitude is {round(magnitude.item(), 3)}; "
        return result_str

    def print_probabilities(self):
        probabilities = torch.sigmoid(self.probabilities.data)
        keys = self.LEARNING_COLOR_LIST + self.LEARNING_STN_LIST
        result_str = ""
        for key, probability in zip(keys, probabilities):
            result_str += f"key is {key}, probability is {round(probability.item(), 3)}; "
        return result_str

    @torch.no_grad()
    def _clamp(self):
        self.probabilities.data = torch.clamp(self.probabilities.data, -10, 10)
        self.magnitudes.data = torch.clamp(self.magnitudes.data, -10, 10)

    def forward(self, image, boxes=None, labels=None, semantic_seg=None):
        self._clamp()
        p = torch.sigmoid(self.probabilities)
        m = torch.sigmoid(self.magnitudes)
        p = relaxed_bernoulli(p)
        _len = p.shape[0]
        index = torch.randperm(_len).to(image.device)
        index = index[: self.solve_number].tolist()
        color_result = []
        p_iter = 0
        m_iter = 0
        for tran in self.learning_color_model_list:
            if p_iter in index:
                _m = m[p_iter].view(-1, 1).expand(image.shape[0], -1)
                now_image, _ = tran(image, _m, boxes)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                color_result.append(now_image - image)
            p_iter += 1
            m_iter += 1

        stn_result = []
        for tran in self.learning_stn_model_list:
            if p_iter in index:
                _m = m[p_iter].view(-1, 1).expand(image.shape[0], -1)
                H = tran(image, _m)
                H = p[p_iter] * H + (1 - p[p_iter]) * torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(H.device).expand_as(H)
                stn_result.append(H - torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(H.device).expand_as(H))
            p_iter += 1
            m_iter += 1

        for tran in self.nolearning_model_list:
            if p_iter in index:
                now_image, _ = tran(image, boxes)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                color_result.append(now_image - image)
            p_iter += 1

        if len(color_result) > 0:
            image = image + torch.stack(color_result).sum(0)

        if len(stn_result) > 0:
            stn_result = torch.stack(stn_result).sum(0) + \
                         torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(stn_result[0].device).expand_as(stn_result[0])
            if semantic_seg == None:
                image, boxes, labels = self.forward_stn_det(image, stn_result, boxes, labels)
            else:
                image, semantic_seg = self.forward_stn_seg(image, stn_result, semantic_seg)
        if semantic_seg == None:
            return image, boxes, labels
        else:
            return image, semantic_seg

    def forward_stn_det(self, x, H, boxes, labels):
        grid = torch.nn.functional.affine_grid(H, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        boxes, labels = self.forward_box(boxes, labels, H, x.shape)
        return x, boxes, labels

    def forward_stn_seg(self, x, H, semantic_seg):
        grid = torch.nn.functional.affine_grid(H, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        semantic_seg = torch.nn.functional.grid_sample(semantic_seg.float() + 1, grid, mode="nearest")
        semantic_seg[semantic_seg == 0] = 255
        semantic_seg = semantic_seg - 1
        semantic_seg[semantic_seg >= 20] = 255
        semantic_seg = semantic_seg.contiguous().long()
        return x, semantic_seg

    def forward_box(self, boxes, labels, H, size):
        b, c, h, w = size
        center_h, center_w = h / 2, w / 2
        result_boxes = []
        result_labels = []
        _w = H[:, 0, 0] * H[:, 1, 1] - H[:, 1, 0] * H[:, 0, 1]
        new00 = H[:, 1, 1] / _w
        new01 = - H[:, 0, 1] / _w
        new02 = (H[:, 0, 2] * H[:, 1, 1] - H[:, 1, 2] * H[:, 0, 1]) / _w
        _w = - _w
        new10 = H[:, 1, 0] / _w
        new11 = - H[:, 0, 0] / _w
        new12 = (H[:, 0, 2] * H[:, 1, 0] - H[:, 1, 2] * H[:, 0, 0]) / _w
        new00, new01, new02, new10, new11, new12 = new00.clone(), new01.clone(), new02.clone(), new10.clone(), new11.clone(), new12.clone()
        H = torch.stack([torch.stack([new00, new10], dim=-1), torch.stack([new01, new11], dim=-1),
                         torch.stack([new02, new12], dim=-1)], dim=-1)
        H = H.contiguous()
        for i, box in enumerate(boxes):  # min_x,min_y,max_x,_max_y
            label = labels[i]
            min_x, min_y, max_x, max_y = torch.split(
                box, 1, dim=-1)
            # [1,1] [-1,1]
            # [1,-1] [-1,-1]
            # xold = xnew * w00 + ynew* w01 + w02
            # yold = xnew * w10 + ynew* w11 + w12
            # => xold*w11 = xnew * w00*w11 + ynew * w01 * w11 + w02 * w11
            # => yold*w01 = xnew * w10*w01 + ynew * w01 * w11 + w12 * w01
            # => ((xold*w11-yold*w01) + w02*w11 - w12*w01 )/(w00*w11-w10*w01) = xnew
            # => w11/(w00*w11-w10*w01) * xold + (-w01)/(w00*w11-w10*w01) + (w02*w11 - w12*w01)/(w00*w11-w10*w01) = xnew
            #
            # => xold*w10 = xnew * w00*w10 + ynew * w01 * w10 + w02 * w10
            # => yold*w00 = xnew * w10*w00+ ynew * w11 * w00 + w12 * w00
            # => xold * w10 - yold * w00 + (w02*w10 - w12 * w00) = ynew * (w01*w10 - w11*w00)
            # => w10/(w01*w10-w11*w00) * xold + (-w00)/(w01*w10-w11*w00) * yold + (w02*w10-w12*w00)/(w01*w10 - w11*w00) = ynew
            min_x, min_y, max_x, max_y = -(min_x - center_w) / center_w, -(min_y - center_h) / center_h, -(
                    max_x - center_w) / center_w, -(max_y - center_h) / center_h
            coordinates = torch.stack([torch.stack([min_x, min_y]), torch.stack([max_x, min_y]),
                                       torch.stack([min_x, max_y]), torch.stack([max_x, max_y])])  # [4, 2, nb_bbox, 1]
            coordinates = torch.cat(
                (coordinates,
                 torch.ones(4, 1, coordinates.shape[2], 1, dtype=coordinates.dtype).to(coordinates.device)),
                dim=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.permute(2, 0, 1, 3)  # [nb_bbox, 4, 3, 1]
            coordinates = torch.matmul(H[i], coordinates)  # [nb_bbox, 4, 2, 1]
            coordinates = coordinates[..., 0]  # [nb_bbox, 4, 2]
            coordinates[:, :, 1] = coordinates[:, :, 1] * -center_h + center_h
            coordinates[:, :, 0] = coordinates[:, :, 0] * -center_w + center_w
            min_x, min_y = torch.min(coordinates[:, :, 0], dim=1)[0], torch.min(coordinates[:, :, 1], dim=1)[0]
            max_x, max_y = torch.max(coordinates[:, :, 0], dim=1)[0], torch.max(coordinates[:, :, 1], dim=1)[0]
            min_x[min_x < 0] = 0
            min_x[min_x > w] = w
            min_y[min_y < 0] = 0
            min_y[min_y > h] = h
            max_x = torch.where(max_x < min_x, min_x, max_x)
            max_x[max_x > w] = w
            max_y = torch.where(max_y < min_y, min_y, max_y)
            max_y[max_y > h] = h

            box = torch.stack([min_x, min_y, max_x, max_y],
                              dim=-1)
            mask = (box[:, 0] != box[:, 2]) & (box[:, 1] != box[:, 3])
            box = box[mask, :]
            label = label[mask]
            if box.shape[0] > 0:
                result_boxes.append(box)
                result_labels.append(label)
            else:
                result_boxes.append(box)
                result_labels.append(label)
                print("[WARNING] box.shape[0]==0")

        return result_boxes, result_labels
