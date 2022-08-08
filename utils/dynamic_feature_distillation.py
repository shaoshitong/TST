import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from utils.cka import kernel_CKA, linear_CKA


class Mlp(nn.Module):
    """
    References:
        x -> fc -> act -> drop -> fc2 -> drop -> y
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
            self,
            dim,
            window_size,
            num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            pretrained_window_size=[0, 0],
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
                .permute(1, 2, 0)
                .contiguous()
                .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
        )

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )
        self.apply(self._init_weights)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):  # B, (H W) C
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x  # B (H * W) C

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(Classifier, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.fc(self.pool(x))
        return x


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class DynamicFeatureDistillation(nn.Module):
    def __init__(
            self,
            features_size: tuple,
            teacher_channels: tuple,
            student_channels: tuple,
            patch_size=4,
            swinblocknumber=[4, 3, 2],
            distill_mode="all",
            num_classes=100,
    ):
        """
        This dynamic knowledge distillation requires that
        the student and instructor resolutions are aligne
        d, but does not require channel alignment.
        """
        super(DynamicFeatureDistillation, self).__init__()
        assert len(features_size) == len(teacher_channels) and len(teacher_channels) == len(
            student_channels
        ), "the student and instructor resolutions must be alignedd, but does not require channel alignment."
        self.features_size = features_size
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        self.swinblocknumber = swinblocknumber
        self.patch_size = patch_size
        assert distill_mode in ["all", "one", "last_two"]
        self.distill_mode = distill_mode

        if self.distill_mode == "one":
            distill_number = -1
        elif self.distill_mode == "last_two":
            distill_number = -2
        else:
            distill_number = 0
        self.distill_number = distill_number
        # TODO: build first embedding conv layer

        self.teacher_first_conv_embeddings = nn.ModuleList([])
        for size, t_channel, s_channel in zip(
                features_size[distill_number:],
                teacher_channels[distill_number:],
                student_channels[distill_number:],
        ):
            conv_layer = nn.Conv2d(
                in_channels=t_channel,
                out_channels=s_channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            )
            self.teacher_first_conv_embeddings.append(conv_layer)
        self.student_first_conv_embeddings = nn.ModuleList([])
        for size, s_channel, s_channel in zip(
                features_size[distill_number:],
                student_channels[distill_number:],
                student_channels[distill_number:],
        ):
            conv_layer = nn.Conv2d(
                in_channels=s_channel,
                out_channels=s_channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            )
            self.student_first_conv_embeddings.append(conv_layer)

        # TODO: apply ViT to embedding mix informations

        self.vit_encoder1_embeddings = nn.ModuleList([])
        ite = 0
        for size, s_channel in zip(
                features_size[distill_number:], student_channels[distill_number:]
        ):
            vit_embedding = nn.Sequential(*[])
            for i in range(self.swinblocknumber[ite]):
                vit_embedding.add_module(
                    f"swin_{i}",
                    SwinTransformerBlock(
                        dim=s_channel,
                        num_heads=4,
                        window_size=patch_size,
                        input_resolution=(size, size),
                        drop_path=0.2,
                        shift_size=0 if (i % 2 == 1) else patch_size // 2,
                    ),
                )
            ite += 1
            self.vit_encoder1_embeddings.append(vit_embedding)

        self.vit_encoder2_embeddings = nn.ModuleList([])
        ite = 0
        for size, s_channel in zip(
                features_size[distill_number:], student_channels[distill_number:]
        ):
            vit_embedding = nn.Sequential(*[])
            for i in range(self.swinblocknumber[ite]):
                vit_embedding.add_module(
                    f"swin_{i}",
                    SwinTransformerBlock(
                        dim=s_channel,
                        num_heads=4,
                        window_size=patch_size,
                        input_resolution=(size, size),
                        drop_path=0.2,
                        shift_size=0 if (i % 2 == 1) else patch_size // 2,
                    ),
                )
            ite += 1
            self.vit_encoder2_embeddings.append(vit_embedding)

        self.vit_decoder_embeddings = nn.ModuleList([])
        ite = 0
        for size, s_channel, t_channel in zip(
                features_size[distill_number:],
                student_channels[distill_number:],
                teacher_channels[distill_number:]
        ):
            vit_embedding = nn.Sequential(*[])
            for i in range(self.swinblocknumber[ite]):
                vit_embedding.add_module(
                    f"swin_{i}",
                    SwinTransformerBlock(
                        dim=s_channel,
                        num_heads=4,
                        window_size=patch_size,
                        input_resolution=(size, size),
                        drop_path=0.2,
                        shift_size=0 if (i % 2 == 1) else patch_size // 2,
                    ),
                )
            ite += 1
            self.vit_decoder_embeddings.append(vit_embedding)

        self.student_unembedding = nn.ModuleList([])
        for size, s_channel, t_channel in zip(
                features_size[distill_number:],
                student_channels[distill_number:],
                teacher_channels[distill_number:]
        ):
            self.student_unembedding.append(nn.Conv2d(s_channel, t_channel, (1, 1), (1, 1), bias=False))

        self.student_bns = nn.ModuleList([
            LayerNorm2d(s_channel) for s_channel in student_channels[distill_number:]
        ])
        self.teacher_bns = nn.ModuleList([
            LayerNorm2d(s_channel) for s_channel in student_channels[distill_number:]
        ])

        self.student_bns2 = nn.ModuleList([
            LayerNorm2d(s_channel) for s_channel in student_channels[distill_number:]
        ])
        self.teacher_bns2 = nn.ModuleList([
            LayerNorm2d(t_channel) for t_channel in teacher_channels[distill_number:]
        ])
        self.student_fcs = nn.ModuleList([
            Classifier(s_channel, num_classes) for s_channel in student_channels[distill_number:]
        ])
        self.teacher_fcs = nn.ModuleList([
            Classifier(s_channel, num_classes) for s_channel in student_channels[distill_number:]
        ])
        self.cross = nn.CrossEntropyLoss()

        # TODO: build flatten

        self.flatten = nn.Flatten()

    def mix_student_and_teacher(
            self, teacher_feature_map, student_feature_map, ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Here, we perform a completely random mask
        """
        b, c, h, w = teacher_feature_map.shape
        soft_mask = torch.rand(1, 1, h, w).to(teacher_feature_map.device)
        hard_mask = soft_mask > ratio
        return torch.where(hard_mask, teacher_feature_map, student_feature_map)

    def vit_forward(self, feature_map, vit_embedding):
        """
        Support both student and teacher's feature map forward communication here
        """
        b, c, h, w = feature_map.shape
        feature_map = rearrange(feature_map, "b c h w -> b (h w) c")
        feature_map = vit_embedding(feature_map)
        feature_map = rearrange(feature_map, "b (h w) c -> b c h w", h=h, w=w)
        return feature_map

    def all_feature_map_vit_forward(self, feature_maps, vit_embeddings):
        result = []
        for feature_map, vit_embedding in zip(feature_maps, vit_embeddings):
            f = self.vit_forward(feature_map, vit_embedding)
            result.append(f)
        return result

    def all_fist_conv_layer_forward(self, feature_maps, conv_embeddings):
        result = []
        for feature_map, conv_embedding in zip(feature_maps, conv_embeddings):
            f = conv_embedding(feature_map)
            # f = rearrange(
            #     f, "b (c p q) h w -> b c (p w) (q h)", p=self.patch_size, q=self.patch_size
            # )
            result.append(f)
        return result

    def compute_ratio(self, teacher_feature_maps, student_feature_maps, if_linear=True):
        result = []

        for teacher_feature_map, student_feature_map in zip(
                teacher_feature_maps, student_feature_maps
        ):
            teacher_feature_map = self.flatten(teacher_feature_map)
            student_feature_map = self.flatten(student_feature_map)
            if if_linear:
                CKA = linear_CKA(student_feature_map, teacher_feature_map)
            else:
                CKA = kernel_CKA(student_feature_map, teacher_feature_map)
            CKA = CKA.item()
            result.append(CKA)

        return result

    def ratio_update(self, ratio):
        if not hasattr(self, "ratios"):
            self.ratios = [0.5 for i in range(len(self.features_size[self.distill_number:]))]
        for i, r in enumerate(ratio):
            self.ratios[i] = 0.9 * self.ratios[i] + 0.1 * r

    def bn_forward(self, feature_maps, bn_embeddings):
        result = []
        for feature_map, bn_embedding in zip(feature_maps, bn_embeddings):
            f = bn_embedding(feature_map)
            result.append(f)
        return result

    def fc_forward(self, feature_maps, fc_embeddings):
        result = []
        for feature_map, fc_embedding in zip(feature_maps, fc_embeddings):
            f = fc_embedding(feature_map)
            result.append(f)
        return result

    def kl_loss(self, teacher_logits, student_logits, targets, temperature=1):
        kl_loss = 0.
        for teacher_logit, student_logit in zip(teacher_logits, student_logits):
            a = (temperature ** 2) * F.kl_div(
                torch.log_softmax(student_logit / temperature, 1),
                torch.softmax(teacher_logit / temperature, 1), reduction="batchmean",
            )
            b = self.cross(teacher_logit, targets)
            kl_loss += (a + b)
        return kl_loss

    def forward(self, teacher_feature_maps, student_feature_maps, targets) -> torch.Tensor:
        teacher_feature_maps = teacher_feature_maps[self.distill_number:]
        student_feature_maps = student_feature_maps[self.distill_number:]

        # TODO: Only original sample
        # b = teacher_feature_maps[0].shape[0] // 2
        # teacher_feature_maps = [i[b:] for i in teacher_feature_maps]
        # student_feature_maps = [i[b:] for i in student_feature_maps]
        # targets = targets[b:]

        student_feature_maps = self.bn_forward(student_feature_maps, self.student_bns2)
        teacher_feature_maps = self.bn_forward(teacher_feature_maps, self.teacher_bns2)


        assert isinstance(teacher_feature_maps, list) and isinstance(student_feature_maps, list)
        assert len(teacher_feature_maps) == len(student_feature_maps)

        student_feature_maps = self.all_fist_conv_layer_forward(
            student_feature_maps, self.student_first_conv_embeddings
        )
        new_teacher_feature_maps = self.all_fist_conv_layer_forward(
            teacher_feature_maps, self.teacher_first_conv_embeddings
        )

        student_feature_maps = self.all_feature_map_vit_forward(
            student_feature_maps, self.vit_encoder1_embeddings
        )
        new_teacher_feature_maps = self.all_feature_map_vit_forward(
            new_teacher_feature_maps, self.vit_encoder2_embeddings
        )


        student_feature_maps = self.bn_forward(student_feature_maps, self.student_bns)
        teacher_feature_maps = self.bn_forward(teacher_feature_maps, self.teacher_bns)


        student_feature_logits = self.fc_forward(student_feature_maps, self.student_fcs)
        teacher_feature_logits = self.fc_forward(teacher_feature_maps, self.teacher_fcs)


        kl_loss = self.kl_loss(teacher_feature_logits, student_feature_logits, targets)
        ratios = self.compute_ratio(new_teacher_feature_maps, student_feature_maps)
        self.ratio_update(ratios)
        ratios = self.ratios
        mix_student_feature_maps = []
        for ratio, new_teacher_feature_map, student_feature_map in zip(
                ratios, new_teacher_feature_maps, student_feature_maps
        ):
            mix_student_feature_map = self.mix_student_and_teacher(
                new_teacher_feature_map, student_feature_map, ratio=ratio
            )
            mix_student_feature_maps.append(mix_student_feature_map)

        student_feature_maps = self.all_feature_map_vit_forward(
            mix_student_feature_maps, self.vit_decoder_embeddings
        )

        student_feature_maps = self.all_fist_conv_layer_forward(student_feature_maps, self.student_unembedding)
        dfd_loss = 0.0

        for teacher_feature_map, student_feature_map in zip(
                teacher_feature_maps, student_feature_maps
        ):
            dfd_loss += F.mse_loss(teacher_feature_map, student_feature_map, reduction="mean")

        return dfd_loss, kl_loss

# if __name__ == "__main__":
#     dpk = DynamicFeatureDistillation(features_size=(32, 16, 8), teacher_channels=(16, 32, 64),
#                                      student_channels=(8, 16, 32)).cuda()
#     T = [torch.randn(2, 16, 32, 32).cuda(), torch.randn(2, 32, 16, 16).cuda(), torch.randn(2, 64, 8, 8).cuda()]
#     S = [torch.randn(2, 8, 32, 32).cuda(), torch.randn(2, 16, 16, 16).cuda(), torch.randn(2, 32, 8, 8).cuda()]
#     loss = dpk(T, S)
#     print(loss)
