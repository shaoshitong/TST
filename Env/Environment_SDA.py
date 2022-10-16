import os
import time

import timm.scheduler.scheduler
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy

from datas.Augmention import Mulit_Augmentation
from helpers.correct_num import correct_num
from helpers.log import Log
from losses.SimpleMseKD import SMSEKD
from pretrain.CIFAR100_color import run_color
from pretrain.CIFAR100_stn import run_stn


def criticion(type, alpha=1, beta=1):
    def ne_ce_loss(student_out, teacher_out, label):
        t_loss = F.cross_entropy(teacher_out, label)
        s_loss = F.cross_entropy(student_out, label)
        return alpha * t_loss - beta * s_loss

    def ne_confidence_ce_loss(student_out, teacher_out, label):
        mask = label.bool()
        weight = 1 - teacher_out.softmax(1)[mask]
        t_loss = F.cross_entropy(teacher_out, label, reduction="none")
        s_loss = F.cross_entropy(student_out, label, reduction="none")
        loss = ((alpha * t_loss - beta * s_loss) * weight).sum() / weight.sum()
        return loss

    if type == "NO_CONFIDENCE":
        return ne_ce_loss
    else:
        return ne_confidence_ce_loss


class conv_relu_bn(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(conv_relu_bn, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (stride, stride), (stride, stride), (0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.bn(self.relu(self.conv(x)))
        return x


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
    def __init__(self, yaml, gpu, teacher):
        self.lr = yaml["SDA"]["lr"]
        self.gpu = gpu
        self.SDA = DDP(
            Mulit_Augmentation(
                pretrain_path=yaml["SDA"]["pretrain_path"],
                dataset_type=yaml["SDA"]["dataset_type"],
                solve_number=yaml["SDA"]["solve_number"],
            ).cuda(gpu),
            device_ids=[gpu],
        )
        self.yaml = yaml
        self.criticion = criticion(yaml["SDA"]["criticion_type"])
        self.optimizer = torch.optim.AdamW(self.SDA.parameters(), lr=self.lr)
        if yaml["SDA"]["finetune_teacher"]:
            self.afe = DDP(AugmentationFeatureEncoder(self.yaml).cuda(gpu), device_ids=[gpu])
            self.optimizer_afe = torch.optim.AdamW(self.afe.parameters(), lr=self.lr, weight_decay=1e-4)

    def __call__(self, student, teacher, x, y, if_learning=True, if_afe=False):
        if if_afe:
            self.step_afe(student, teacher, x, y, if_learning)
            augment_x, augment_y = x, y
        else:
            augment_x, augment_y = self.step(student, teacher, x, y, if_learning)
        return augment_x, augment_y, self.loss

    def step(self, student, teacher, x, y, if_learning):
        if not self.yaml["only_stage_one"] and if_learning:
            student.eval()
            student.requires_grad_(False)
            augment_x = x.clone()
            augment_y = y.clone()
            augment_x.requires_grad = True
            augment_x = self.SDA(augment_x)
            student_out = student(augment_x)
            teacher_tuple, teacher_out = teacher(augment_x, is_feat=True)
            teacher_tuple = teacher_tuple[:-1]
            if self.yaml["SDA"]["finetune_teacher"]:
                teacher_out = self.afe(teacher_tuple)
            loss = self.criticion(student_out, teacher_out, augment_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.loss = loss.item()
            self.optimizer.step()
            # give back
            student.train()
            student.requires_grad_(True)
        else:
            self.loss = 0.0

        with torch.no_grad():
            augment_x = self.SDA(x.clone())
            augment_y = y.clone()

        return augment_x.detach(), augment_y.detach()

    def step_afe(self, student, teacher, x, y, if_learning):
        if not self.yaml["only_stage_one"] and if_learning:
            student.eval()
            student.requires_grad_(False)
            augment_x = x.clone()
            augment_y = y.clone()
            augment_x.requires_grad = True
            augment_x = self.SDA(augment_x)
            teacher_tuple, teacher_out = teacher(augment_x, is_feat=True)
            teacher_tuple = teacher_tuple[:-1]
            if self.yaml["SDA"]["finetune_teacher"]:
                teacher_out = self.afe(teacher_tuple)
            loss = F.cross_entropy(teacher_out, augment_y)
            self.optimizer_afe.zero_grad()
            loss.backward()
            self.loss = loss.item()
            self.optimizer_afe.step()
            # give back
            student.train()
            student.requires_grad_(True)
        else:
            self.loss = 0.0


class LearnDiversifyEnv(object):
    def __init__(
            self,
            dataloader: DataLoader,
            testloader: DataLoader,
            student_model: nn.Module,
            teacher_model: nn.Module,
            scheduler: torch.optim.lr_scheduler.MultiStepLR,
            optimizer: torch.optim.Optimizer,
            loss: nn.Module,
            yaml,
            wandb,
            gpu,
    ):
        super(LearnDiversifyEnv, self).__init__()
        # TODO: basic settings
        self.dataloader = dataloader
        self.testloader = testloader
        self.student_model = student_model
        self.teacher_model = teacher_model
        assert isinstance(self.dataloader.dataset, torch.utils.data.Dataset)
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.num_classes = yaml["num_classes"]
        self.loss = loss
        self.epoch = 0
        self.begin_epoch = 0
        self.total_epoch = yaml["epoch"]
        self.yaml = yaml
        self.wandb = wandb
        self.model_save_path = yaml["model_save_path"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.gpu = gpu
        self.done = False
        self.best_acc = 0.0
        self.accumuate_count = 0
        self.scaler = torch.cuda.amp.GradScaler()
        if self.gpu == 0:
            self.log = Log(log_each=yaml["log_each"])
        time_path = time.strftime("%Y^%m^%d^%H^%M^%S", time.localtime()) + ".txt"
        if self.gpu == 0:
            self.ff = open(time_path, "w")

        # TODO: Learning to diversify
        self.pretrain()
        print("pretrain finished successfully!")
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad = False

        self.convertor = SDAGenerator(yaml=yaml, gpu=gpu, teacher=self.teacher_model)
        # TODO: It is important to remember to add the last parameter in the optimizer
        self.weights = yaml["weights"]
        self.only_satge_one = self.yaml["only_stage_one"]
        self.convertor_training_epoch = self.yaml["SDA"]["convertor_training_epoch"]
        self.convertor_epoch_number = self.yaml["SDA"]["convertor_epoch_number"]

        if yaml["resume"] != "none":
            dict = torch.load(yaml["resume"])
            self.optimizer.load_state_dict(dict["optimizer"])
            self.scheduler.load_state_dict(dict["scheduler"])
            self.student_model.load_state_dict(dict["student_model"])
            self.convertor.SDA.load_state_dict(dict["convertor"])
            self.scaler.load_state_dict(dict["scaler"])
            self.best_acc = dict["acc"]
            self.begin_epoch = self.epoch = dict["epoch"] + 1
            print(f"successfully load checkpoint from {yaml['resume']}")

    def KDLoss(self, student_output, teacher_output, targets=None, temperature=4):
        soft_loss = F.kl_div(
            torch.log_softmax(student_output / temperature, dim=1),
            torch.softmax(teacher_output / temperature, dim=1),
            reduction="batchmean",
        )
        hard_loss = (
            F.kl_div(torch.log_softmax(student_output, dim=-1), targets, reduction="batchmean")
            if targets != None
            else 0.0
        )
        return hard_loss + (temperature ** 2) * soft_loss * self.yaml["criticion"]["alpha"]

    def run_one_train_batch_size(self, batch_idx, indexs, input, target):
        input = input.float().cuda(self.gpu)
        target = target.cuda(self.gpu)
        target = target.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).float()
        # TODO: Learning to diversify
        with torch.cuda.amp.autocast(enabled=True):
            inputs_max, target_temp, ne_ce_loss = self.convertor(
                self.student_model, self.teacher_model, input, target, False
            )
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])
        b, c, h, w = data_aug.shape
        with torch.cuda.amp.autocast(enabled=True):
            (student_tuple, student_logits) = self.student_model(data_aug, is_feat=True)
            with torch.no_grad():
                (teacher_tuple, teacher_logits) = self.teacher_model(data_aug, is_feat=True)
                if self.yaml["SDA"]["finetune_teacher"]:
                    for i in range(len(teacher_tuple)):
                        teacher_tuple[i] = teacher_tuple[i][:b // 2]
                    teacher_logits[:b // 2] = self.convertor.afe(teacher_tuple[:-1])
        # TODO: compute relative loss
        # TODO: 1, vanilla KD Loss
        vanilla_kd_loss = self.KDLoss(
            student_logits.float(),
            teacher_logits.float(),
            labels,
            self.yaml["criticion"]["temperature"],
        )
        # TODO: 2. Combine all Loss in stage one
        loss = (self.weights[0] * vanilla_kd_loss)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # TODO: Compute top1 and top5
        top1, top5 = correct_num(student_logits[: input.shape[0]], target, topk=(1, 5))
        dist.all_reduce(top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(top5, op=dist.ReduceOp.SUM)
        top1 /= torch.cuda.device_count()
        top5 /= torch.cuda.device_count()
        lr = (
            self.scheduler.get_last_lr()
            if hasattr(self.scheduler, "get_lr")
            else self.scheduler.lr()
        )
        lr = lr[0]
        if self.gpu == 0:
            self.log(self.teacher_model, (loss + ne_ce_loss).cpu(), top1.cpu(), top5.cpu(), lr)
        return (
            top1.cpu().item(),
            vanilla_kd_loss.cpu().item(),
            ne_ce_loss,
        )

    def run_one_convertor_batch_size(self, batch_idx, indexs, input, target, if_afe):
        input = input.float().cuda(self.gpu)
        target = target.cuda(self.gpu)
        target = target.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).float()
        # TODO: Learning to diversify
        with torch.cuda.amp.autocast(enabled=True):
            inputs_max, target_temp, ne_ce_loss = self.convertor(
                self.student_model, self.teacher_model, input, target, True, if_afe
            )
        if batch_idx == 0:
            from utils.save_Image import change_tensor_to_image
            change_tensor_to_image(inputs_max[0], "image", f"cifar100_{self.epoch}")
        return (
            ne_ce_loss,
        )

    @torch.no_grad()
    def run_one_val_batch_size(self, input, target):
        input = input.float().cuda()
        target = target.cuda()
        target = target.view(-1)
        logits = self.student_model(input).float()
        t_logits = self.teacher_model(input)
        loss = self.loss(logits, t_logits, target)
        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1 = dist.all_reduce(top1, op=dist.ReduceOp.SUM) / torch.cuda.device_count()
        return top1.cpu().item(), loss.cpu().item()

    def run_one_convertor_epoch(self, if_afe):
        self.student_model.train()
        self.convertor.SDA.train()
        total_ne_ce_loss = 0
        for batch_idx, (index, input, target) in enumerate(self.dataloader):
            (
                ne_ce_loss,
            ) = self.run_one_convertor_batch_size(batch_idx, index, input, target, if_afe)
            if self.gpu == 0:
                self.wandb.log(
                    {
                        "ne_ce_loss": ne_ce_loss,
                        **{f"p_{i}": p for i, p in
                           enumerate(self.convertor.SDA.module.probabilities.data.clone().detach().sigmoid().tolist())},
                        **{f"m_{i}": m for i, m in
                           enumerate(self.convertor.SDA.module.magnitudes.data.clone().detach().sigmoid().tolist())},
                    },
                    step=self.accumuate_count,
                )
            total_ne_ce_loss = total_ne_ce_loss + ne_ce_loss
            self.accumuate_count += 1
        total_ne_ce_loss = total_ne_ce_loss / len(self.dataloader)
        if self.gpu == 0:
            self.ff.write(
                f"epoch:{self.epoch}, ne_ce_loss:{total_ne_ce_loss}\n"
            )
            print(f"when train {'AFE' if if_afe else 'SDA'}, ne_ce_loss is: {total_ne_ce_loss}")

    def run_one_train_epoch(self):
        """============================train=============================="""
        start_time = time.time()
        self.student_model.train()
        self.convertor.SDA.train()
        if self.gpu == 0:
            self.log.train(len_dataset=len(self.dataloader))

        # TODO: DIVERSIFY LEARNING
        if self.epoch in self.convertor_training_epoch:
            for i in range(int(self.convertor_epoch_number/2)):
                self.run_one_convertor_epoch(True)
            for i in range(int(self.convertor_epoch_number/2)):
                self.run_one_convertor_epoch(False)

        for batch_idx, (index, input, target) in enumerate(self.dataloader):
            (
                top1,
                vanilla_kd_loss,
                ne_ce_loss,
            ) = self.run_one_train_batch_size(batch_idx, index, input, target)
            if self.gpu == 0:
                self.wandb.log(
                    {
                        "top1": top1,
                        "vanilla_kd_loss": vanilla_kd_loss,
                        "ne_ce_loss": ne_ce_loss,
                        **{f"p_{i}": p for i, p in
                           enumerate(self.convertor.SDA.module.probabilities.data.clone().detach().sigmoid().tolist())},
                        **{f"m_{i}": m for i, m in
                           enumerate(self.convertor.SDA.module.magnitudes.data.clone().detach().sigmoid().tolist())},
                    },
                    step=self.accumuate_count,
                )
            self.accumuate_count += 1
        if self.gpu == 0:
            train_acc, train_loss = (
                self.log.epoch_state["top_1"] / self.log.epoch_state["steps"],
                self.log.epoch_state["loss"] / self.log.epoch_state["steps"],
            )
        else:
            train_acc, train_loss = 0, 0
        use_time = round((time.time() - start_time) / 60, 2)
        if self.gpu == 0:
            self.ff.write(
                f"epoch:{self.epoch}, train_acc:{train_acc}, train_loss:{train_loss}, min:{use_time}\n"
            )
        return train_acc, train_loss

    @torch.no_grad()
    def run_one_val_epoch(self):
        """============================val=============================="""
        start_time = time.time()
        self.student_model.eval()
        self.convertor.SDA.eval()
        if self.gpu == 0:
            self.log.eval(len_dataset=len(self.testloader))
        for batch_idx, (input, target) in enumerate(self.testloader):
            input = input.float().cuda()
            target = target.cuda()
            input.requires_grad_()
            torch.cuda.synchronize()
            logits = self.student_model(input)
            torch.cuda.synchronize()
            loss = F.cross_entropy(logits, target, reduction="mean")
            top1, top5 = correct_num(logits, target, topk=(1, 5))
            dist.all_reduce(top1, op=dist.ReduceOp.SUM)
            dist.all_reduce(top5, op=dist.ReduceOp.SUM)
            top1 /= torch.cuda.device_count()
            top5 /= torch.cuda.device_count()
            if self.gpu == 0:
                self.log(self.student_model, loss.cpu(), top1.cpu(), top5.cpu())
        if self.gpu == 0:
            test_acc, test_loss = (
                self.log.epoch_state["top_1"] / self.log.epoch_state["steps"],
                self.log.epoch_state["loss"] / self.log.epoch_state["steps"],
            )
        else:
            test_acc, test_loss = 0, 0
        use_time = round((time.time() - start_time) / 60, 2)
        if self.gpu == 0:
            self.ff.write(
                f"epoch:{self.epoch}, test_acc:{test_acc}, test_loss:{test_loss}, min:{use_time}\n"
            )
        return test_acc

    def scheduler_step(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
            self.scheduler.step()
        elif isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
            self.scheduler.step(self.epoch)

    def pretrain(self):
        run_color(self.yaml)
        run_stn(self.yaml)

    def training_in_all_epoch(self):

        for i in range(self.begin_epoch, self.total_epoch):
            self.dataloader.sampler.set_epoch(i)
            ttop1, tloss = self.run_one_train_epoch()
            vtop1 = self.run_one_val_epoch()
            self.scheduler_step()
            if self.gpu == 0:
                self.wandb.log(
                    {"train_loss": tloss, "train_top1": ttop1, "val_top1": vtop1}, step=self.epoch
                )
            self.epoch += 1
            if self.best_acc < vtop1:
                self.best_acc = vtop1
            path = self.model_save_path
            if self.gpu == 0:
                if not os.path.isdir(path):
                    os.makedirs(path)
                if i == self.total_epoch - 1:
                    model_path = os.path.join(
                        self.model_save_path,
                        f"final_dataset_{self.yaml['data']}_teacher_{self.yaml['tarch']}_student_{self.yaml['arch']}.pth",
                    )
                else:
                    model_path = os.path.join(
                        self.model_save_path,
                        f"epoch_{i}_dataset_{self.yaml['data']}_teacher_{self.yaml['tarch']}_student_{self.yaml['arch']}.pth",
                    )
                dict = {
                    "epoch": self.epoch,
                    "acc": vtop1,
                    "scaler": self.scaler.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "student_model": self.student_model.state_dict(),
                    "convertor": self.convertor.SDA.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }
                torch.save(dict, model_path)

        if self.gpu == 0:
            self.log.flush()
            self.ff.close()
            self.wandb.finish()


"""
for name,param in model.named_parameters():
    if param.grad is not None:
        print(name)
"""
