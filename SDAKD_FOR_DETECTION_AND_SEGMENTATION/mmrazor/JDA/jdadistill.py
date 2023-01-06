import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.algorithms.general_distill import GeneralDistill
from mmrazor.models import ALGORITHMS
from tkinter import _flatten

@ALGORITHMS.register_module()
class JDADistill(GeneralDistill):
    def __init__(self,
                 collect_key,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):
        self.collect_key = collect_key
        super(JDADistill, self).__init__(with_student_loss, with_teacher_loss, **kwargs)

    def train_step(self, data, optimizer):
        for key in self.collect_key:
            if key == "img":
                b, n = data[key].shape[:2]
                data[key] = data[key].contiguous().view(b * n, *data[key].shape[2:])
            else:
                data[key] = list(_flatten(data[key]))
        data["img_metas"] = list(_flatten(data["img_metas"]))
        return super(JDADistill, self).train_step(data, optimizer)
