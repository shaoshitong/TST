import copy
import math

import einops
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
    Tuple,
)
from datas.Augmentation import cutmix


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.normalize(x, x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6, inplace=False)
        return x


class Flow_Attention(nn.Module):
    # flow attention in normal version
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=1e-6):
        super(Flow_Attention, self).__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        ## 1. Linear projection
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (
            torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps)
        )
        source_outgoing = 1.0 / (
            torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps)
        )
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum(
            "nhld,nhd->nhl",
            queries + self.eps,
            (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps,
        )
        conserved_source = torch.einsum(
            "nhld,nhd->nhl",
            keys + self.eps,
            (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps,
        )
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(
            conserved_sink * (float(queries.shape[2]) / float(keys.shape[2]))
        )
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        # (4) dot product
        x = (
                self.dot_product(
                    queries * sink_incoming[:, :, :, None],  # for value normalization
                    keys,
                    values * source_competition[:, :, :, None],
                )  # competition
                * sink_allocation[:, :, :, None]
        ).transpose(
            1, 2
        )  # allocation
        ## (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
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
        self.policies = [
            ("AutoContrast", p, None),
            ("Contrast", p, 3),
            ("Posterize", p, 0),
            ("Solarize", p, 4),
            ("TranslateY", p, 8),
            ("ShearX", p, 5),
            ("Brightness", p, 3),
            ("ShearY", p, 0),
            ("TranslateX", p, 1),
            ("Sharpness", p, 5),
            ("Invert", p, None),
            ("Color", p, 4),
            ("Equalize", p, None),
            ("Rotate", p, 3),
        ] if policy == AutoAugmentPolicy.CIFAR10 else self.policies
        self.policies.append(('CutMix', None, None))
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
        self.fc.add_module("fc1", nn.Linear(len(self.policies) * C * H * W, 512))
        self.fc.add_module("relu", nn.ReLU(inplace=True))
        self.fc.add_module("fc2", nn.Linear(512, len(self.policies)))
        self.p = p
        for param in list(list(self.fc.parameters())):
            param.requires_grad = True

        self.set_buffer()

    def set_buffer(self):
        number_of_samples = self.num_train_samples
        number_of_policies = len(self.policies)
        self.buffer = torch.ones(number_of_samples, number_of_policies).cuda().float()

    def buffer_update(self, indexs, weight, epoch):
        """
        indexs: [bs,]
        logits: [bs,num_classes]
        """
        # momentum = epoch / (epoch + 1)
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
        assert isinstance(img, Tensor), "The input must be Tensor!"
        assert (
                img.shape[1] == 1 or img.shape[1] == 3
        ), "The channels for image input must be 1 and 3"
        if img.dtype != torch.uint8:
            if self.policy == AutoAugmentPolicy.CIFAR10:
                img.mul_(torch.Tensor([0.2675, 0.2565, 0.2761])[None, :, None, None].cuda()).add_(
                    torch.Tensor([0.5071, 0.4867, 0.4408])[None, :, None, None].cuda()
                )
            else:
                img.mul_(torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()).add_(
                    torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()
                )
            img = img * 255
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

        randperm = torch.arange(len(self.policies))
        img_size = F.get_image_size(img)
        op_meta = self._augmentation_space(10, img_size)
        # TODO: 让每种操作都进行，所以模型该学习何物？
        results = []
        lasbels = [y]
        results.append(self.tran(img / 255))
        # TODO: 应当用竞争机制来生成对应的输出...
        for randindex in randperm:
            prob = torch.rand((1,))
            sign = torch.randint(2, (1,))
            policy = self.policies[randindex]
            (op_name, p, magnitude_id) = policy
            p = self.p
            if prob <= p:
                if op_name != "CutMix":
                    magnitudes, signed = op_meta[op_name]
                    magnitude = (
                        float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                    )
                    if signed and sign:
                        magnitude *= -1.0
                    img = _apply_op(
                        img, op_name, magnitude, interpolation=self.interpolation, fill=fill
                    )
                else:
                    img, y = cutmix(img, y, num_classes=y.shape[1])
            results.append(self.tran(img / 255))
            lasbels.append(y)
        results = torch.stack(results, 0)  # P,B,C,H,W
        labels = torch.stack(lasbels, 0)
        P, B, C, H, W = results.shape
        results = results.view(P, B, -1)  # P,B,C*H*W
        # TODO: 使用注意力机制来生成权重，为了计算计算量，我可以使用flowformer?
        # TODO: 在这里，注意力机制的Batchsize维度应该是第二维度，第一维度才是要注意的地方。
        # TODO: 但问题在于Flowfromer的输出是要保证和输入value相同的，这点他做不到，实际上我们希望对所有的pixel信息进行编码，或许可以借鉴SKattention?

        attention_vector = (
                einops.rearrange(
                    torch.sigmoid(self.fc(einops.rearrange(results[1:], "p b c -> b (p c)"))),
                    "b c -> c b",
                )[..., None]
                + 1
        )
        attention_vector = attention_vector[randperm].contiguous()  # P,B,1
        attention_vector = attention_vector / (attention_vector.sum(0)) * attention_vector.shape[0]

        # TODO: 解决数值不稳定的问题
        self.buffer_update(indexs, attention_vector[..., 0].permute(1, 0), epoch)
        use_attention_vector = self.buffer[indexs].permute(1, 0)[..., None]
        if epoch % 2 == 0:
            attention_vector = use_attention_vector.detach()
        else:
            attention_vector = attention_vector
        attention_vector = torch.ones_like(attention_vector, device=attention_vector.device)
        # TODO: End
        x0 = attention_vector[0]  # 1,B,1
        different_vector = attention_vector - torch.cat(
            [attention_vector[1:], attention_vector[0].unsqueeze(0)], 0
        )
        different_vector[-1] = attention_vector[
            -1
        ]  # TODO:可逆矩阵推导，a1=x1-x2,a2=x2-x3,...,an-1=xn-1-xn,an=xn
        result = (
                (different_vector * results[1:]).sum(0) + (1 - x0) * results[0].unsqueeze(0)
        ).view(B, C, H, W)
        labels = (
                (different_vector * labels[1:]).sum(0) + (1 - x0) * labels[0].unsqueeze(0)
        ).view(B, -1)
        return result, labels

#
# model = LearningAutoAugment(policy=AutoAugmentPolicy.CIFAR10, C=3, H=32, W=32, alpha=0.0).cuda()
# begin = PIL.Image.open("/home/sst/product/RLDCD/output/original28_sample.png").convert('RGB')
# image = transforms.ToTensor()(begin)
# print(image.max(), image.min())
# now_image = transforms.Compose([transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])(
#     image).unsqueeze(0)
# begin.show()
# output = model(now_image.cuda())
# output.mul_(torch.Tensor([0.2675, 0.2565, 0.2761])[None, :, None, None].cuda()).add_(
#                     torch.Tensor([0.5071, 0.4867, 0.4408])[None, :, None, None].cuda())
# output = output * 255
# torch.clip_(output, 0, 255)
# output = output.type(torch.uint8).cpu()
# now_output = transforms.ToPILImage()(output[0])
# now_output.show()
# A是使用randperm,B是使用AA的p
