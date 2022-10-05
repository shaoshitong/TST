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


class Reshape(nn.Module):
    def __init__(self, C, H, W, P):
        super(Reshape, self).__init__()
        self.conv = nn.AvgPool2d(kernel_size=(7,7),stride=(7,7))
        self.P = P

    def forward(self, x):
        p, b, c, h, w = x.shape
        x = einops.rearrange(
            self.conv(einops.rearrange(x, "p b c h w -> (p b) c h w")),
            "(p b) c h w -> b (p c h w)",
            p=p,
        )
        return x


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
        # TODO: 重建对应所有的算子
        self.policies_set = []
        self.C = C
        self.H = H
        self.W = W
        self.num_train_samples = num_train_samples
        self.tag = 0
        self.total_epoch = total_epoch
        all_policies_set = set()
        for policies in self.policies:
            first_policies = policies[0]
            second_policies = policies[1]
            if first_policies[0] not in all_policies_set:
                self.policies_set.append(copy.deepcopy(first_policies))
                all_policies_set.add(first_policies[0])
            if second_policies[0] not in all_policies_set:
                self.policies_set.append(copy.deepcopy(second_policies))
                all_policies_set.add(second_policies[0])
        self.policies = list(self.policies_set)
        self.policies.append(("CutMix", None, None))
        print(self.policies)
        self.tran = (
            transforms.Compose(
                [transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            )
            if policy == AutoAugmentPolicy.CIFAR10
            else transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        # TODO: Learning Module
        self.fc = nn.Sequential()
        if H > 56 and W > 56:
            self.fc.add_module("conv1", Reshape(C=C, H=H, W=W, P=len(self.policies)+1))
            H, W = H // 7, W // 7
        self.fc.add_module("fc1", nn.Linear((len(self.policies)+1)* C * H * W, 512))
        self.fc.add_module("relu", nn.ReLU(inplace=True))
        self.fc.add_module("fc2", nn.Linear(512, len(self.policies)+1))
        self.p = p
        for param in list(list(self.fc.parameters())):
            param.requires_grad = True

        self.set_buffer()

    def set_buffer(self):
        number_of_samples = self.num_train_samples
        number_of_policies = len(self.policies) + 1
        self.buffer = torch.ones(number_of_samples, number_of_policies).cuda().float()

    def buffer_update(self, indexs, weight, epoch):
        """
        indexs: [bs,]
        logits: [bs,num_classes]
        """
        momentum = 0.9

        self.buffer[indexs] = (
            self.buffer[indexs]
            .mul_(momentum)
            .add_((1.0 - momentum) * weight.clone().detach().float())
        )

    def forward(self, img: Tensor, y, indexs, epoch):
        """
        Tensor -> Tensor (to translate)
        """
        randperm = torch.arange(len(self.policies)+1)
        with torch.no_grad():
            assert isinstance(img, Tensor), "The input must be Tensor!"
            assert (
                img.shape[1] == 1 or img.shape[1] == 3
            ), "The channels for image input must be 1 and 3"

            if epoch % 2 == 0:
                if img.dtype != torch.uint8:
                    if self.policy == AutoAugmentPolicy.CIFAR10:
                        img.mul_(
                            torch.Tensor([0.2675, 0.2565, 0.2761])[None, :, None, None].cuda()
                        ).add_(torch.Tensor([0.5071, 0.4867, 0.4408])[None, :, None, None].cuda())
                    else:
                        img.mul_(
                            torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
                        ).add_(torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda())
                    img = img * 255
                    img = torch.floor(img + 0.5)
                    torch.clip_(img, 0, 255)
                    img = img.type(torch.uint8)
                assert (
                    img.dtype == torch.uint8
                ), "Only torch.uint8 image tensors are supported, but found torch.int64"

                fill = self.fill
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * F.get_image_num_channels(img)
                elif fill is not None:
                    fill = [float(f) for f in fill]
                img_size = F.get_image_size(img)
                op_meta = self._augmentation_space(10, img_size)
                # TODO: 让每种操作都进行，所以模型该学习何物？
                results = []
                labels = y
                results.append(self.tran(img / 255))
                b = img.shape[0]
                # TODO: 应当用竞争机制来生成对应的输出...
                for randindex in randperm[:-1]:
                    prob = torch.rand((1,))
                    sign = torch.randint(2, (1,))
                    policy = self.policies[randindex]
                    (op_name, p, magnitude_id) = policy
                    p = self.p
                    if op_name != "CutMix":
                        magnitudes, signed = op_meta[op_name]
                        magnitude = (
                            float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                        )
                        if signed and sign:
                            magnitude *= -1.0
                        if int(p * b) > 0:
                            index = torch.randperm(b)[:int(p*b)].to(img.device)
                            img[index] = _apply_op(
                                img[index], op_name, magnitude, interpolation=self.interpolation, fill=fill
                            )
                    else:
                        if prob <= p:
                            img, labels = cutmix(img, y, num_classes=y.shape[1])

                    results.append(self.tran(img / 255))

                results = torch.stack(results, 0)  # P,B,C,H,W

                self.register_buffer("results", results)
                self.register_buffer("labels", labels)
            else:
                results = self.results
                labels = self.labels

        # TODO: 使用注意力机制来生成权重，为了计算计算量，我可以使用flowformer?
        # TODO: 在这里，注意力机制的Batchsize维度应该是第二维度，第一维度才是要注意的地方。
        # TODO: 但问题在于Flowfromer的输出是要保证和输入value相同的，这点他做不到，实际上我们希望对所有的pixel信息进行编码，或许可以借鉴SKattention?
        results.requires_grad = True
        labels.requires_grad = True
        P, B, C, H, W = results.shape
        if self.policy == AutoAugmentPolicy.CIFAR10:
            results = results.view(P, B, -1)  # P,B,C*H*W
            attention_vector = (
                einops.rearrange(
                    torch.sigmoid(self.fc(einops.rearrange(results, "p b c -> b (p c)"))),
                    "b c -> c b",
                )[..., None]
                + 1
            )
        else:
            attention_vector = (
                einops.rearrange(
                    torch.sigmoid(self.fc(results)),
                    "b c -> c b",
                )[..., None]
                + 1
            )
            results = results.view(P, B, -1)  # P,B,C*H*W
        attention_vector = attention_vector[randperm].contiguous()  # P,B,1
        attention_vector = attention_vector / (attention_vector.sum(0)) * attention_vector.shape[0]

        # TODO: 解决数值不稳定的问题
        # self.buffer_update(indexs, attention_vector[..., 0].permute(1, 0), epoch)
        # use_attention_vector = self.buffer[indexs].permute(1, 0)[..., None]
        # if epoch % 2 == 0:
        #     attention_vector = use_attention_vector.detach()
        # else:
        #     attention_vector = attention_vector
        # TODO: End
        x0 = attention_vector[0]  # 1,B,1
        different_vector = attention_vector - torch.cat(
            [attention_vector[1:], attention_vector[0].unsqueeze(0)], 0
        )
        different_vector[-1] = attention_vector[
            -1
        ]  # TODO:可逆矩阵推导，a1=x1-x2,a2=x2-x3,...,an-1=xn-1-xn,an=xn
        result = (
            (different_vector * results).sum(0)
        ).view(B, C, H, W)

        if (indexs == 10).sum().item() != 0:
            from utils.save_Image import  change_tensor_to_image
            change_tensor_to_image(result[0],"images",f"m_{epoch}")
        return result, labels, attention_vector.mean(1).squeeze()
