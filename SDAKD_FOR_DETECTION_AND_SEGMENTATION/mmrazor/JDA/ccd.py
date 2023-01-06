from mmrazor.models.builder import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F


@LOSSES.register_module()
class CCD(nn.KLDivLoss):
    def __init__(self, temperature, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)
        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.momentum = 0.99

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        b1_indices = torch.arange(y_s.shape[0]) % 2 == 0
        b2_indices = torch.arange(y_s.shape[0]) % 2 != 0
        original_soft_loss = super().forward(torch.log_softmax(y_s[b1_indices] / self.temperature, dim=1),
                                             torch.softmax(y_t[b1_indices] / self.temperature, dim=1))
        b1 = y_t[b1_indices]
        b2 = y_t[b2_indices]
        cosine = F.cosine_similarity(b1, b2) + 1
        augmented_soft_loss = self.kl_loss(torch.log_softmax(y_s[b2_indices] / self.temperature, dim=1),
                                           torch.softmax(y_t[b2_indices] / self.temperature, dim=1)) * cosine.unsqueeze(
            -1)
        augmented_soft_loss = augmented_soft_loss.sum(-1).mean()
        soft_loss = (original_soft_loss + augmented_soft_loss) / 2
        return soft_loss
