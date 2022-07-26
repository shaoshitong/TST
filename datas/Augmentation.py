import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as transforms


@torch.no_grad()
def cutmix(
        x: torch.tensor, y: torch.tensor, cutmix_prob: int = 0.1, beta: int = 0.3, num_classes: int = 60
) -> torch.tensor:
    if np.random.rand() > cutmix_prob:
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=num_classes).float()
        return x, y
    N, _, H, W = x.shape
    indices = torch.randperm(N, device=torch.device("cuda"))
    label = torch.zeros((N, num_classes), device=torch.device("cuda"))
    x1 = x[indices, :, :, :]
    lam = np.random.beta(beta, beta)
    rate = np.sqrt(1 - lam)
    cut_x, cut_y = int((H * rate) // 2), int((W * rate) // 2)
    if cut_x == H // 2 or cut_y == W // 2:
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=num_classes).float()
        return x, y
    if y.ndim > 1 and y.shape[1] != 1:
        y = y.argmax(1)
    y1 = y.clone()[indices]
    cx, cy = int(np.random.randint(cut_x, H - cut_x)), int(np.random.randint(cut_y, W - cut_x))
    bx1, bx2 = cx - cut_x, cx + cut_x
    by1, by2 = cy - cut_y, cy + cut_y
    x[:, :, bx1:bx2, by1:by2] = x1[:, :, bx1:bx2, by1:by2].clone()
    label[torch.arange(N), y] = lam
    label[torch.arange(N), y1] = 1 - lam
    return x, label


def yoco(x: torch.Tensor, trans: callable):
    def YOCO(images, aug, h, w):
        images = torch.cat((aug(images[:, :, :, 0:int(w / 2)]), aug(images[:, :, :, int(w / 2):w])), dim=3) if \
            torch.rand(1) > 0.5 else torch.cat((aug(images[:, :, 0:int(h / 2), :]), aug(images[:, :, int(h / 2):h, :])),
                                               dim=2)
        return images

    return YOCO(x, trans(x), x.shape[2], x.shape[3])


class Augmentation(nn.Module):
    def __init__(self, num_classes, policy="cifar10", mode=['autoaugment']):
        super(Augmentation, self).__init__()
        if policy == "cifar10":
            use_policy = transforms.AutoAugmentPolicy.CIFAR10
        elif policy == 'imagenet':
            use_policy = transforms.AutoAugmentPolicy.IMAGENET
        else:
            raise NotImplementedError
        self.autoaugment = transforms.AutoAugment(use_policy)
        self.cutmix = cutmix
        self.yoco = yoco
        self.mode = mode
        self.num_classes = num_classes
        self.tran = transforms.Compose([transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]) if policy=='cifar10' else \
                    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    @torch.no_grad()
    def forward(self, x, y):
        if 'autoaugment' in self.mode and 'yoco' not in self.mode:
            x.mul_(torch.Tensor([0.5071, 0.4867, 0.4408])[:, None, None].cuda()).add_(
                torch.Tensor([0.2675, 0.2565, 0.2761])[:, None, None].cuda())
            x = x * 255
            torch.clip_(x, 0, 255)
            x = x.type(torch.uint8)
            x = self.autoaugment(x)
            x = self.tran(x / 255)
        elif 'yoco' in self.mode:
            x.mul_(torch.Tensor([0.5071, 0.4867, 0.4408])[:, None, None].cuda()).add_(
                torch.Tensor([0.2675, 0.2565, 0.2761])[:, None, None].cuda())
            x = x * 255
            torch.clip_(x, 0, 255)
            x = x.type(torch.uint8)
            x = self.yoco(x, self.autoaugment)
            x = self.tran(x / 255)

        if 'cutmix' in self.mode:
            x, y = self.cutmix(x, y, num_classes=self.num_classes)
        return x, y
