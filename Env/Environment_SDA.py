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


def criticion(type):
    def ne_ce_loss(student_out, teacher_out, label, alpha=1, beta=1):
        t_loss = F.cross_entropy(teacher_out, label)
        s_loss = F.cross_entropy(student_out, label)
        return alpha * t_loss - beta * s_loss

    def ne_confidence_ce_loss(student_out, teacher_out, label, alpha=1, beta=1):
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
        )
        self.yaml = yaml
        self.criticion = criticion(yaml["SDA"]["criticion_type"])
        self.optimizer = torch.optim.SGD(self.SDA.parameters(), lr=self.lr, momentum=0.9)

    def __call__(self, student, teacher, x, y):
        augment_x, augment_y = self.step(student, teacher, x, y)
        return augment_x, augment_y, self.loss

    def step(self, student, teacher, x, y):
        if not self.yaml["only_stage_one"]:
            self.SDA.requires_grad_(True)
            student.eval()
            student.requires_grad_(False)
            augment_x = x.clone()
            augment_y = y.clone()
            augment_x.requires_grad = True
            augment_x = self.SDA(augment_x)
            loss = self.criticion(student(augment_x), teacher(augment_x), augment_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.loss = loss.item()
            self.optimizer.step()
            # give back
            self.SDA.requires_grad_(False)
            student.train()
            student.requires_grad_(True)
        else:
            self.loss = 0.0
        with torch.no_grad():
            augment_x = self.SDA(x.clone())
            augment_y = y.clone()

        return augment_x.detach(), augment_y.detach()


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
        self.convertor = SDAGenerator(yaml=yaml, gpu=gpu)
        # TODO: It is important to remember to add the last parameter in the optimizer
        self.weights = yaml["weights"]

        # TODO: DFD
        # self.dfd = DDP(DynamicFeatureDistillation(
        #     features_size=yaml["dfd"]["feature_size"],
        #     teacher_channels=yaml["dfd"]["teacher_channels"],
        #     student_channels=yaml["dfd"]["student_channels"],
        #     patch_size=yaml["dfd"]["patch_size"],
        #     distill_mode=yaml["dfd"]["distill_mode"],
        #     swinblocknumber=yaml["dfd"]["swinblocknumber"],
        #     mode=yaml['dfd']['mode'],
        # ).cuda(gpu),device_ids=[gpu])

        self.dfd = DDP(
            SMSEKD(
                in_channels=yaml["dfd"]["student_channels"],
                out_channels=yaml["dfd"]["teacher_channels"],
                shapes=yaml["dfd"]["feature_size"],
                out_shapes=yaml["dfd"]["feature_size"],
                num_classes=yaml["num_classes"],
            ).cuda(gpu),
            device_ids=[gpu],
        )

        self.optimizer.add_param_group({"params": self.dfd.parameters()})
        self.only_satge_one = self.yaml["only_stage_one"]

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

        # TODO: turn target (N,1) -> (N,C)
        input = input.float().cuda(self.gpu)
        target = target.cuda(self.gpu)
        target = target.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).float()

        # TODO: Learning to diversify
        with torch.cuda.amp.autocast(enabled=True):
            inputs_max, target_temp, ne_ce_loss = self.convertor(
                self.student_model, self.teacher_model, input, target
            )

        if batch_idx == 0:
            from utils.save_Image import change_tensor_to_image

            change_tensor_to_image(inputs_max[0], "image", f"cifar100_{self.epoch}")
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])

        with torch.cuda.amp.autocast(enabled=True):
            (student_tuple, student_logits) = self.student_model(data_aug, is_feat=True)
            with torch.no_grad():
                (teacher_tuple, teacher_logits) = self.teacher_model(data_aug, is_feat=True)
            student_tuple = student_tuple[:-1]
            teacher_tuple = teacher_tuple[:-1]

        # TODO: compute relative loss

        # TODO: 1, vanilla KD Loss
        vanilla_kd_loss = self.KDLoss(
            student_logits.float(),
            teacher_logits.float(),
            labels,
            self.yaml["criticion"]["temperature"],
        )

        # TODO: 2 DFD Loss
        if self.weights[1] == 0:
            dfd_loss = torch.Tensor([0.0]).cuda(self.gpu)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                dfd_loss = self.dfd(student_tuple, teacher_tuple, labels, only_alignment=True)

        # TODO: 3. Combine all Loss in stage one
        loss = (self.weights[0] * vanilla_kd_loss + self.weights[1] * dfd_loss)
        # self.convertor_optimizer.zero_grad()
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(self.dfd.parameters(), max_norm=2, norm_type=2)
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
            dfd_loss.cpu().item(),
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

    def run_one_train_epoch(self):
        """============================train=============================="""
        start_time = time.time()
        self.student_model.train()
        self.convertor.SDA.train()
        if self.gpu == 0:
            self.log.train(len_dataset=len(self.dataloader))
        for batch_idx, (index, input, target) in enumerate(self.dataloader):
            (
                top1,
                vanilla_kd_loss,
                ne_ce_loss,
                dfd_loss,
            ) = self.run_one_train_batch_size(batch_idx, index, input, target)
            if self.gpu == 0:
                self.wandb.log(
                    {
                        "top1": top1,
                        "vanilla_kd_loss": vanilla_kd_loss,
                        "ne_ce_loss": ne_ce_loss,
                        "dfd_loss": dfd_loss,
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
                    "dfd": self.dfd.state_dict(),
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
