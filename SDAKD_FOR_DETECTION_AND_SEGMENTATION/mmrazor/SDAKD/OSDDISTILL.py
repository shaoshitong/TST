import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.algorithms.general_distill import GeneralDistill
from mmrazor.models import ALGORITHMS
from tkinter import _flatten
from .DeAug import Mulit_Augmentation
from .DV import detection_vis
from torch.nn.parallel import DistributedDataParallel as DDP
import os,copy
from mmrazor.models.utils import add_prefix
from mmcv.runner.hooks import OptimizerHook

@ALGORITHMS.register_module()
class OSDDistill(GeneralDistill):
    def __init__(self,
                 collect_key,
                 solve_number,
                 convertor_training_epoch,
                 convertor_epoch_number,
                 pretrain_path,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):
        super(OSDDistill, self).__init__(with_student_loss, with_teacher_loss, **kwargs)
        self.collect_key = collect_key
        self.solve_number = solve_number
        self.convertor_training_epoch = convertor_training_epoch
        gpu = int(os.environ['LOCAL_RANK'])
        self.gpu = gpu
        self.convertor_epoch_number = convertor_epoch_number
        self.convertor = DDP(Mulit_Augmentation(
            pretrain_path=pretrain_path,
            solve_number=self.solve_number
        ).cuda(gpu),
            device_ids=[gpu],
            find_unused_parameters=True )
        self._iter = 0

    def set_convertor(self,modules:nn.Module):
        if not isinstance(modules,nn.Module):
            return
        if hasattr(modules,"set_convertor_training"):
            modules.set_convertor_training()
            print(f"successfully set convertor training in {modules.__class__.__name__}")
        for name,module in modules.named_children():
            self.set_convertor(module)

    def unset_convertor(self,modules:nn.Module):
        if not isinstance(modules,nn.Module):
            return
        if hasattr(modules,"unset_convertor_training"):
            modules.unset_convertor_training()
            print(f"successfully unset convertor training in {modules.__class__.__name__}")
        for name,module in modules.named_children():
            self.unset_convertor(module)

    def train_convertor_step(self, data, optimizer):
        augment_data = dict()
        if self.gpu==0:
            data["img"] = data["img"][0,...][None,...]
            if "gt_semantic_seg" in data.keys():
                data["gt_semantic_seg"] = data["gt_semantic_seg"][0, ...][None, ...]
            else:
                data["gt_bboxes"] = [data["gt_bboxes"][0]]
                data["gt_labels"] = [data["gt_labels"][0]]
            data["img_metas"] = [data["img_metas"][0]]
        augment_data["img_metas"] = copy.deepcopy(data["img_metas"])
        data["img"].requires_grad = True
        if "gt_semantic_seg" in data.keys():
            augment_data["img"], augment_data["gt_semantic_seg"] = self.convertor(data["img"],semantic_seg=data["gt_semantic_seg"])
        else:
            augment_data["img"] ,augment_data["gt_bboxes"] ,augment_data["gt_labels"]\
                = self.convertor(data["img"] ,boxes = data["gt_bboxes"] , labels = data["gt_labels"])
        losses = dict()

        # TODO: TRAINING STUDENT
        student_losses = self.distiller.exec_student_forward(
            self.architecture, augment_data)
        student_losses = add_prefix(student_losses, 'student')
        losses.update(student_losses)

        # TODO: TRAINING TEACHER
        teacher_losses = self.distiller.exec_teacher_forward(augment_data)
        teacher_losses = add_prefix(teacher_losses, 'teacher')
        losses.update(teacher_losses)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(augment_data['img'].data))
        return outputs

    def train_step(self, data, optimizer):
        if "gt_semantic_seg" in data.keys():
            gt_semantic_seg = data["gt_semantic_seg"]
            img = data["img"]
            with torch.no_grad():
                new_img, new_gt_semantic_seg = self.convertor.module(img, semantic_seg = gt_semantic_seg)
            img = torch.cat([new_img, img], dim=0)
            gt_semantic_seg = torch.cat([new_gt_semantic_seg.clone(),gt_semantic_seg],dim=0)
            data["img"] = img
            data["gt_semantic_seg"] = gt_semantic_seg
        else:
            gt_bboxes = data["gt_bboxes"]
            gt_labels = data["gt_labels"]
            img = data["img"]
            with torch.no_grad():
                new_img, new_gt_bboxes, new_gt_labels = self.convertor.module(img,boxes = gt_bboxes,labels = gt_labels)
            img = torch.cat([new_img,img],dim=0)
            gt_bboxes = new_gt_bboxes + gt_bboxes
            gt_labels = new_gt_labels + gt_labels
            data["gt_bboxes"] = gt_bboxes
            data["gt_labels"] = gt_labels
            data["img"] = img
        data["img_metas"] = data["img_metas"] + copy.deepcopy(data["img_metas"])
        self._iter+=1
        return super(OSDDistill, self).train_step(data, optimizer)

