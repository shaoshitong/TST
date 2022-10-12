import copy
import math

import einops,random
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import (
    AutoAugmentPolicy,
    InterpolationMode,
    List,
    Optional,
    Tensor,
)

from datas.Augmentation import cutmix
from .SelectGAN import TranslateNet

class LearningAutoAugment(transforms.AutoAugment):
    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        p=0.25,
        C=3,
        H=224,
        W=224,
        num_train_samples=50000,
        total_epoch=240,
    ):
        super(LearningAutoAugment, self).__init__(
            policy,
            interpolation,
            fill,
        )

        if policy == AutoAugmentPolicy.IMAGENET:
            self.translate_gen = TranslateNet(type="imagenet")
        else:
            self.translate_gen = TranslateNet(type="cifar100")

    def forward(self, img: Tensor, y, indexs, epoch,estimation=False):
        """
        Tensor -> Tensor (to translate)
        """
        result = self.translate_gen(img)
        labels = y
        attention_vector = torch.ones(14,1).to(result.device)
        return result, labels, attention_vector.mean(1).squeeze()
