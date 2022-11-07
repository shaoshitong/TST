from datas.pretrain.CIFAR100_color import run_cifar100_color
from datas.pretrain.CIFAR100_stn import run_cifar100_stn
from datas.pretrain.ImageNet_color import run_imagenet_color
from datas.pretrain.ImageNet_stn import run_imagenet_stn
from datas.Augmention import Mulit_Augmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def criticion(type, alpha=1, beta=1):
    def ne_ce_loss(student_out, teacher_out, label):
        t_loss = F.cross_entropy(teacher_out, label)
        s_loss = F.cross_entropy(student_out, label)
        return alpha * t_loss, - beta * s_loss
    return ne_ce_loss

class conv_relu_bn(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(conv_relu_bn, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (stride, stride), (stride, stride), (0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.bn(self.relu(self.conv(x)))
        return x


class ALRS():
    '''
    proposer: Huanran Chen
    theory: landscape
    Bootstrap Generalization Ability from Loss Landscape Perspective
    '''

    def __init__(self, optimizer, loss_threshold=0.02, loss_ratio_threshold=0.02, decay_rate=0.9):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and abs(delta / (self.last_loss - 1e-6)) < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.decay_rate
                now_lr = group['lr']
                print(f'now lr = {now_lr}')

        self.last_loss = loss


class AugmentationFeatureEncoder(nn.Module):
    def __init__(self, yaml):
        super(AugmentationFeatureEncoder, self).__init__()
        self.yaml = yaml
        self.channels = yaml["dfd"]["teacher_channels"]
        self.shapes = yaml["dfd"]["feature_size"]
        self.num_classes = yaml["num_classes"]
        self.embedding_model_list = nn.ModuleList([])

        l = len(self.channels)
        for i in range(l - 1):
            in_channel = self.channels[i]
            out_channel = self.channels[i + 1]
            in_shape = self.shapes[i]
            out_shape = self.shapes[i + 1]
            stride = int(in_shape // out_shape)
            layer = conv_relu_bn(in_channel, out_channel, stride)
            self.embedding_model_list.append(layer)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.channels[-1], self.num_classes)
        )

    def forward(self, feature_tuple):
        out_feature = self.embedding_model_list[0](feature_tuple[0])
        out_feature = feature_tuple[1] + out_feature
        out_feature = self.embedding_model_list[1](out_feature)
        out_feature = feature_tuple[2] + out_feature
        return self.classifier(out_feature)

class SDAGenerator:
    def __init__(self, yaml, gpu):
        self.lr = yaml["SDA"]["lr"]
        self.gpu = gpu
        self.SDA = DDP(
            Mulit_Augmentation(
                pretrain_path=yaml["SDA"]["pretrain_path"],
                dataset_type=yaml["SDA"]["dataset_type"],
                solve_number=yaml["SDA"]["solve_number"],
            ).cuda(gpu),
            device_ids=[gpu],
            find_unused_parameters=True if yaml["SDA"]["solve_number"] <= 2 else False
        )
        self.yaml = yaml
        self.criticion = criticion(yaml["SDA"]["criticion_type"])
        self.optimizer = torch.optim.SGD(self.SDA.parameters(),
                                         lr=0.01,
                                         momentum=0.9)
        self.scheduler = ALRS(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_classes = yaml["num_classes"]
        self.pretrain()

    def reset(self):
        del self.scaler
        del self.optimizer
        del self.scheduler
        self.optimizer = torch.optim.SGD(self.SDA.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = ALRS(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

    def __call__(self, student, teacher, x, y, if_learning=True, if_afe=False):
        augment_x, augment_y = self.step(student, teacher, x, y, if_learning)
        return augment_x, augment_y, self.loss_s,self.loss_t

    def step(self, student, teacher, x, y, if_learning):
        self.loss_t = 0
        self.loss_s = 0

        if not self.yaml["only_stage_one"] and if_learning:
            student.eval()
            student.requires_grad_(False)
            augment_x = x.clone()
            augment_y = y.clone()
            augment_x.requires_grad = True
            augment_x = self.SDA(augment_x)
            student_out = student(augment_x)
            if 'convnext' in self.yaml['tarch'] or 'swin' in self.yaml['tarch']:
                teacher_tuple, teacher_out = teacher(augment_x)
            else:
                teacher_tuple, teacher_out = teacher(augment_x, is_feat=True)
            loss_t,loss_s = self.criticion(student_out, teacher_out, augment_y)
            self.optimizer.zero_grad()
            loss = loss_s+ loss_t
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.SDA.parameters(),10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.loss = loss.item()
            self.loss_t = loss_t.item()
            self.loss_s = loss_s.item()
            self.optimizer.step()
            # give back
            student.train()
            student.requires_grad_(True)
        else:
            self.loss = 0.0
        if not if_learning:
            with torch.no_grad():
                augment_x = self.SDA.module(x.clone())
                augment_y = y.clone()
        return augment_x.detach(), augment_y.detach()

    def quick_step(self, input, target, teacher_model, student_model):
        if target.ndim == 2 and target.shape[1] == 1:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        # TODO: Learning to diversify
        with torch.cuda.amp.autocast(enabled=True):
            inputs_max, target_temp, ne_ce_s_loss,ne_ce_t_loss = self(
                student_model, teacher_model, input, target, True, False
            )
        return (
            ne_ce_s_loss,ne_ce_t_loss
        )

    def quick_epoch(self, dataloader, teacher_model, student_model,mixup_fn=None):
        student_model.train()
        self.SDA.train()
        total_ne_t_ce_loss = 0
        total_ne_s_ce_loss = 0
        total_sample = 0
        for batch_idx, (samples, targets) in enumerate(dataloader):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            samples, targets = mixup_fn(samples, targets)
            (
                ne_t_ce_loss,
                ne_s_ce_loss
            ) = self.quick_step(samples, targets, teacher_model, student_model)
            total_ne_t_ce_loss += (ne_t_ce_loss * samples.shape[0])
            total_ne_s_ce_loss += (ne_s_ce_loss * samples.shape[0])
            total_sample += samples.shape[0]
        total_ne_t_ce_loss = (total_ne_t_ce_loss / total_sample)
        total_ne_s_ce_loss = (total_ne_s_ce_loss / total_sample)

        self.scheduler.step(total_ne_s_ce_loss+total_ne_t_ce_loss)
        print(f"In an epoch, teacher ce loss is {total_ne_t_ce_loss}", f"student ce loss is {total_ne_s_ce_loss}")


    def quick_multi_epoch(self, dataloader, teacher_model, student_model,epoch_number = 1,mixup_fn=None):
        self.reset()
        for i in range(epoch_number):
            self.quick_epoch(dataloader,teacher_model,student_model,mixup_fn)

    def pretrain(self):
        if self.yaml["SDA"]["dataset_type"] == "CIFAR":
            run_cifar100_stn(self.yaml)
            run_cifar100_color(self.yaml)
        else:
            run_imagenet_stn(self.yaml)
            run_imagenet_color(self.yaml)
