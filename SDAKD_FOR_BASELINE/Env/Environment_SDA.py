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

from datas.DistillforLargeModel import mixup
from datas.SDAGAN import SDAGenerator
from helpers.correct_num import correct_num
from helpers.log import Log
from losses.DISTKD import DIST
from utils.ema import ModelEMA


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
        if "convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]:
            self.mixup = mixup()
        self.scaler = torch.cuda.amp.GradScaler()
        if self.gpu == 0:
            self.log = Log(log_each=yaml["log_each"])
        time_path = time.strftime("%Y^%m^%d^%H^%M^%S", time.localtime()) + ".txt"
        if self.gpu == 0:
            self.ff = open(time_path, "w")

        # TODO: Learning to diversify
        print("pretrain finished successfully!")
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad = False

        self.convertor = SDAGenerator(yaml=yaml, gpu=gpu)
        self.weights = yaml["weights"]
        self.only_satge_one = self.yaml["only_stage_one"]
        self.convertor_training_epoch = self.yaml["SDA"]["convertor_training_epoch"]
        self.convertor_epoch_number = self.yaml["SDA"]["convertor_epoch_number"]

        if "ema_update" in self.yaml and self.yaml["ema_update"] == True:
            self.ema_model = ModelEMA(self.student_model, decay=0.9999)
            print("successfully build ema model")
        else:
            self.ema_model = None

        if yaml["resume"] != "none":
            dict = torch.load(yaml["resume"])
            self.optimizer.load_state_dict(dict["optimizer"])
            self.scheduler.load_state_dict(dict["scheduler"])
            self.student_model.load_state_dict(dict["student_model"])
            if "ema_update" in self.yaml and self.yaml["ema_update"] == True:
                self.ema_model.load_state_dict(dict["ema_model"])
            self.convertor.SDA.load_state_dict(dict["convertor"])
            self.scaler.load_state_dict(dict["scaler"])
            self.best_acc = dict["acc"]
            self.begin_epoch = self.epoch = dict["epoch"]
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
        if "convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]:
            return hard_loss + 2 * (temperature ** 2) * soft_loss * self.yaml["criticion"]["alpha"]
        else:
            return hard_loss + (temperature ** 2) * soft_loss * self.yaml["criticion"]["alpha"]

    def DISTLoss(self, student_output, teacher_output, targets=None, temperature=4):
        b = student_output.shape[0]
        original_soft_loss = (
            F.kl_div(
                torch.log_softmax(student_output[b // 2 :] / temperature, dim=1),
                torch.softmax(teacher_output[b // 2 :] / temperature, dim=1),
                reduction="batchmean",
            )
            * (temperature ** 2)
        )
        original_hard_loss = (
            F.kl_div(
                torch.log_softmax(student_output[b // 2 :], dim=-1),
                targets[b // 2 :],
                reduction="batchmean",
            )
            if targets != None
            else 0.0
        )
        augment_soft_loss = DIST(beta=1, gamma=1)(
            student_output[: b // 2], teacher_output[: b // 2]
        )
        return original_hard_loss / 2 + augment_soft_loss / 2 + original_soft_loss / 2

    def run_one_train_batch_size(self, batch_idx, indexs, input, target):
        input = input.cuda(self.gpu, non_blocking=True)
        target = target.cuda(self.gpu, non_blocking=True)
        target = target.view(-1)
        if ("convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]) and input.shape[
            0
        ] % 2 == 0:
            input, target = self.mixup(input, target)

        else:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        # TODO: Learning to diversify
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            inputs_max, target_temp, ne_s_ce_loss, ne_t_ce_loss = self.convertor(
                self.student_model, self.teacher_model, input, target, False
            )
        ne_ce_loss = ne_s_ce_loss + ne_t_ce_loss
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])
        t_data_aug = data_aug
        b, c, h, w = data_aug.shape
        with torch.cuda.amp.autocast(enabled=True):
            (student_tuple, student_logits) = self.student_model(data_aug, is_feat=True)
            with torch.no_grad():
                if "convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]:
                    (teacher_tuple, teacher_logits) = self.teacher_model.module(t_data_aug)
                else:
                    (teacher_tuple, teacher_logits) = self.teacher_model.module(
                        t_data_aug, is_feat=True
                    )
                # TODO: compute relative loss
                # print((teacher_logits.argmax(1)==labels.argmax(1)).sum().item()/student_logits.shape[0])
        # TODO: 1, vanilla KD Loss
        vanilla_kd_loss = self.KDLoss(
            student_logits.float(),
            teacher_logits.float(),
            labels,
            self.yaml["criticion"]["temperature"],
        )

        aug_stduent_logits_confidence = (
            student_logits[: b // 2].softmax(1)[target.bool()].mean().item()
        )
        aug_teacher_logits_confidence = (
            teacher_logits[: b // 2].softmax(1)[target.bool()].mean().item()
        )
        # TODO: 2. Combine all Loss in stage one
        loss = self.weights[0] * vanilla_kd_loss
        if self.accumuate_count % self.yaml["accumulate_step"] == 0:
            # print(f"Accumulate Count Is {self.accumuate_count}, Zero Grad")
            self.optimizer.zero_grad()
        self.scaler.scale(loss / self.yaml["accumulate_step"]).backward()
        if (self.accumuate_count + 1) % self.yaml["accumulate_step"] == 0:
            if "convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]:
                self.scaler.unscale_(
                    self.optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if "ema_update" in self.yaml and self.yaml["ema_update"] == True:
            self.ema_model.update(self.student_model)
        # TODO: Compute top1 and top5
        top1, top5 = correct_num(student_logits[: input.shape[0]], target, topk=(1, 5))
        dist.all_reduce(top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(top5, op=dist.ReduceOp.SUM)
        top1 /= torch.cuda.device_count()
        top5 /= torch.cuda.device_count()
        lr = self.get_lr()
        # TODO: update CosineLRScheduler
        if isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
            self.scheduler.step(self.accumuate_count)
        if self.gpu == 0:
            self.log(self.teacher_model, (loss + ne_ce_loss).cpu(), top1.cpu(), top5.cpu(), lr)
        return (
            top1.cpu().item(),
            vanilla_kd_loss.cpu().item(),
            aug_teacher_logits_confidence,
            aug_stduent_logits_confidence,
            ne_ce_loss,
        )

    def run_one_convertor_batch_size(self, batch_idx, indexs, input, target, if_afe):
        input = input.float().cuda(self.gpu)
        target = target.cuda(self.gpu)
        target = target.view(-1)
        if ("convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]) and input.shape[
            0
        ] % 2 == 0:
            input, target = self.mixup(input, target)
        else:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        # TODO: Learning to diversify
        with torch.cuda.amp.autocast(enabled=True):
            inputs_max, target_temp, ne_ce_s_loss, ne_ce_t_loss = self.convertor(
                self.student_model, self.teacher_model, input, target, True, if_afe
            )
        return (ne_ce_s_loss + ne_ce_t_loss,)

    @torch.no_grad()
    def run_one_val_batch_size(self, input, target):
        input = input.float().cuda()
        target = target.cuda()
        target = target.view(-1)
        logits = self.student_model(input).float()
        _, t_logits = self.teacher_model(input)
        loss = self.loss(logits, t_logits, target)
        top1, top5 = correct_num(logits, target, topk=(1, 5))
        top1 = dist.all_reduce(top1, op=dist.ReduceOp.SUM) / torch.cuda.device_count()
        return top1.cpu().item(), loss.cpu().item()

    def run_one_convertor_epoch(self, if_afe):
        self.student_model.train()
        self.convertor.SDA.train()
        total_ne_ce_loss = 0
        for batch_idx, (index, input, target) in enumerate(self.dataloader):
            (ne_ce_loss,) = self.run_one_convertor_batch_size(
                batch_idx, index, input, target, if_afe
            )
            if self.gpu == 0:
                self.wandb.log(
                    {
                        "ne_ce_loss": ne_ce_loss,
                        "aug_s_con": self.convertor.aug_stduent_logits_confidence,
                        "aug_t_con": self.convertor.aug_teacher_logits_confidence,
                        **{
                            f"p_{i}": p
                            for i, p in enumerate(
                                self.convertor.SDA.module.probabilities.data.clone()
                                .detach()
                                .sigmoid()
                                .tolist()
                            )
                        },
                        **{
                            f"m_{i}": m
                            for i, m in enumerate(
                                self.convertor.SDA.module.magnitudes.data.clone()
                                .detach()
                                .sigmoid()
                                .tolist()
                            )
                        },
                    },
                    step=self.accumuate_count,
                )
            total_ne_ce_loss = total_ne_ce_loss + ne_ce_loss
            self.accumuate_count += 1
        total_ne_ce_loss = total_ne_ce_loss / len(self.dataloader)
        self.convertor.scheduler.step(total_ne_ce_loss)
        if self.gpu == 0:
            self.ff.write(f"epoch:{self.epoch}, ne_ce_loss:{total_ne_ce_loss}\n")
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
            self.convertor.reset()
            for i in range(int(self.convertor_epoch_number)):
                self.run_one_convertor_epoch(False)

        total_aug_s_con = 0.0
        total_aug_t_con = 0.0
        for batch_idx, (index, input, target) in enumerate(self.dataloader):
            (
                top1,
                vanilla_kd_loss,
                aug_t_con,
                aug_s_con,
                ne_ce_loss,
            ) = self.run_one_train_batch_size(batch_idx, index, input, target)
            if self.gpu == 0:
                self.wandb.log(
                    {
                        "top1": top1,
                        "vanilla_kd_loss": vanilla_kd_loss,
                        "ne_ce_loss": ne_ce_loss,
                        "aug_s_con": aug_s_con,
                        "aug_t_con": aug_t_con,
                        **{
                            f"p_{i}": p
                            for i, p in enumerate(
                                self.convertor.SDA.module.probabilities.data.clone()
                                .detach()
                                .sigmoid()
                                .tolist()
                            )
                        },
                        **{
                            f"m_{i}": m
                            for i, m in enumerate(
                                self.convertor.SDA.module.magnitudes.data.clone()
                                .detach()
                                .sigmoid()
                                .tolist()
                            )
                        },
                    },
                    step=self.accumuate_count,
                )
                total_aug_s_con += aug_s_con * input.shape[0]
                total_aug_t_con += aug_t_con * input.shape[0]

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
                f"epoch:{self.epoch}, train_acc:{train_acc}, train_loss:{train_loss},aug_t_con:{total_aug_t_con / len(self.dataloader.dataset)}, aug_s_con:{total_aug_s_con / len(self.dataloader.dataset)}, min:{use_time}\n"
            )
        return train_acc, train_loss

    @torch.no_grad()
    def run_one_val_epoch(self, if_teacher=False):
        """============================val=============================="""
        start_time = time.time()
        if if_teacher:
            self.teacher_model.eval()
        else:
            self.student_model.eval()
        self.convertor.SDA.eval()
        if self.gpu == 0:
            self.log.eval(len_dataset=len(self.testloader))
        for batch_idx, (input, target) in enumerate(self.testloader):
            input = input.float().cuda()
            target = target.cuda()
            torch.cuda.synchronize()
            if if_teacher:
                if "convnext" in self.yaml["tarch"] or "swin" in self.yaml["tarch"]:
                    _, logits = self.teacher_model.module(input)
                else:
                    logits = self.teacher_model.module(input)

            else:
                if "ema_update" in self.yaml and self.yaml["ema_update"] == True:
                    _, logits = self.ema_model.module(input, is_feat=True)
                else:
                    _, logits = self.student_model.module(input, is_feat=True)
            torch.cuda.synchronize()
            loss = F.cross_entropy(logits, target, reduction="mean")
            top1, top5 = correct_num(logits, target, topk=(1, 5))
            dist.all_reduce(top1, op=dist.ReduceOp.SUM)
            dist.all_reduce(top5, op=dist.ReduceOp.SUM)
            top1 /= torch.cuda.device_count()
            top5 /= torch.cuda.device_count()
            if self.gpu == 0:
                if if_teacher:
                    self.log(self.teacher_model, loss.cpu(), top1.cpu(), top5.cpu())
                else:
                    self.log(self.student_model, loss.cpu(), top1.cpu(), top5.cpu())
        if self.gpu == 0:
            test_top1_acc, test_top5_acc, test_loss = (
                self.log.epoch_state["top_1"] / self.log.epoch_state["steps"],
                self.log.epoch_state["top_5"] / self.log.epoch_state["steps"],
                self.log.epoch_state["loss"] / self.log.epoch_state["steps"],
            )
        else:
            test_top1_acc, test_top5_acc, test_loss = 0, 0, 0
        use_time = round((time.time() - start_time) / 60, 2)
        if self.gpu == 0:
            if if_teacher:
                print("Teacher's Top-1 Acc is", test_top1_acc, "%", "Top-5 Acc is", test_top5_acc)
            else:
                print("Student's Top-1 Acc is", test_top1_acc, "%", "Top-5 Acc is", test_top5_acc)
                self.ff.write(
                    f"epoch:{self.epoch}, test_top1_acc:{test_top1_acc}, test_top1_acc:{test_top5_acc}, min:{use_time}\n"
                )
        return test_top1_acc

    def scheduler_step(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
            self.scheduler.step()
        elif isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
            self.scheduler.step(self.epoch)

    def get_lr(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
            lr = self.scheduler.get_last_lr()[0]
        elif isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
            lr = self.scheduler.get_update_values(self.epoch)[0]
        return lr

    def training_in_all_epoch(self):
        """
        EVAL TEACHER FIRST
        """
        if self.yaml["eval_only"] == True:
            self.run_one_val_epoch(if_teacher=False)
            self.begin_epoch = self.total_epoch
        for i in range(self.begin_epoch, self.total_epoch):
            self.dataloader.sampler.set_epoch(i)
            ttop1, tloss = self.run_one_train_epoch()
            vtop1 = self.run_one_val_epoch()
            if not isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
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
                if "ema_update" in self.yaml and self.yaml["ema_update"] == True:
                    dict["ema_model"] = self.ema_model.state_dict()
                torch.save(dict, model_path)

        if self.gpu == 0:
            self.log.flush()
            self.ff.close()
            self.wandb.finish()
