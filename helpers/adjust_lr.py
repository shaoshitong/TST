from bisect import bisect_right

import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_lr(optimizer, epoch, yaml, step=0, all_iters_per_epoch=0):
    cur_lr = 0.0
    if epoch < yaml["warmup_epoch"]:
        cur_lr = (
            yaml["optimizer"]["lr"]
            * float(1 + step + epoch * all_iters_per_epoch)
            / (yaml["warmup_epoch"] * all_iters_per_epoch)
        )
    else:
        epoch = epoch - yaml["warmup_epoch"]
        cur_lr = yaml["optimizer"]["lr"] * 0.1 ** bisect_right(
            yaml["scheduler"]["milestones"], epoch
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr
    return cur_lr
