import os
import time

import numpy as np
import timm.scheduler.scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datas.Augmentation import Augmentation
from helpers.adjust_lr import adjust_lr
from helpers.correct_num import correct_num
from helpers.log import Log
from utils.augnet import BigImageAugNet, SmallImageAugNet
from utils.mmd import conditional_mmd_rbf
from utils.save_Image import change_tensor_to_image


def log_backward(module, grad_inputs, grad_outputs):
    print("=========")
    for grad_input in grad_inputs:
        print(grad_input.norm())
    for grad_output in grad_outputs:
        print(grad_output.norm())
    print(module)
    print("=========")


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
        device=None,
    ):
        super(LearnDiversifyEnv, self).__init__()
        # TODO: basic settings
        self.dataloader = dataloader
        self.testloader = testloader
        assert isinstance(self.dataloader.dataset, torch.utils.data.Dataset)
        if device != None:
            self.student_model = student_model.to(device)
            self.teacher_model = teacher_model.to(device)
        else:
            self.student_model = student_model
            self.teacher_model = teacher_model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.num_classes = yaml["num_classes"]
        self.loss = loss
        self.epoch = 0
        self.total_epoch = yaml["epoch"]
        self.yaml = yaml
        self.wandb = wandb
        self.model_save_path = yaml["model_save_path"]
        self.log = Log(log_each=yaml["log_each"])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.done = False
        self.episode = 0.0
        self.best_acc = 0.0
        self.accumuate_count = 0
        self.scaler = torch.cuda.amp.GradScaler()
        time_path = time.strftime("%Y^%m^%d^%H^%M^%S", time.localtime()) + ".txt"
        self.ff = open(time_path, "w")

        # TODO: Learning to diversify
        AugNet = (
            BigImageAugNet if not yaml["augnettype"] == "SmallImageAugNet" else SmallImageAugNet
        )
        self.convertor = (
            AugNet(img_size=yaml["img_size"], yaml=yaml).cuda()
            if torch.cuda.is_available()
            else AugNet(img_size=yaml["img_size"], yaml=yaml).cpu()
        )
        self.convertor_optimizer = torch.optim.SGD(
            self.convertor.parameters(), lr=yaml["sc_lr"], momentum=0.9
        )
        self.convertor_scheduler = getattr(torch.optim.lr_scheduler, yaml["scheduler"]["type"])(
            self.convertor_optimizer,
            milestones=yaml["scheduler"]["milestones"],
            gamma=yaml["scheduler"]["gamma"],
        )
        self.tran = (
            transforms.Compose(
                [transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            )
            if yaml["augmentation_policy"] == "cifar10"
            else transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.avgpool2d = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.p_logvar = (
            nn.Sequential(
                nn.Linear(self.student_model.last_channel, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
            ).cuda()
            if torch.cuda.is_available()
            else nn.Sequential(
                nn.Linear(self.student_model.last_channel, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
            )
        )
        self.p_mu = (
            nn.Sequential(
                nn.Linear(self.student_model.last_channel, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
            ).cuda()
            if torch.cuda.is_available()
            else nn.Sequential(
                nn.Linear(self.student_model.last_channel, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
            )
        )
        self.reset_parameters(self.p_mu)
        self.reset_parameters(self.p_logvar)
        # TODO: It is important to remember to add the last parameter in the optimizer
        self.optimizer.add_param_group({"params": self.avgpool2d.parameters()})
        self.optimizer.add_param_group({"params": self.p_logvar.parameters()})
        self.optimizer.add_param_group({"params": self.p_mu.parameters()})
        self.weights = yaml["weights"]

        # TODO: Natural Aumentation (i.e., AutoAugment, Cutmix, YOCO)
        self.augmentation = Augmentation(
            num_classes=yaml["num_classes"],
            policy=yaml["augmentation_policy"],
            mode=yaml["augmentation_mode"],
        )
        self.augmented_ratio = yaml["augmented_ratio"]

    def reset_parameters(self, modules):
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight.data, 0, 0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def reparametrize(self, mu, logvar, factor=0.2):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()  # 0-1 gaussian distribution
        ratio = factor * std * eps
        return mu + ratio  # mu - mean , ratio - std

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
        return hard_loss + (temperature ** 2) * soft_loss

    def Contextual(self, a, b):
        acosineb = 1 - F.cosine_similarity(a, b, 1, 1e-8) + 1e-8  # (b,)
        distance = (a.pow(2).sum(-1).sqrt() - b.pow(2).sum(-1).sqrt()).abs() + acosineb
        return -torch.log(distance).mean()

    def ConLoss(self, features, labels, temperature=0.07):
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        b, c = features.shape[0], features.shape[1]
        assert features.shape[0] == labels.shape[0], "batch size should be same"
        labels = labels.contiguous().view(-1, 1)  # (bs,1)
        mask = torch.eq(labels, labels.T).float().to(features.device)  # (bs,bs)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 2*bs,embedding_dim
        dot_contrast = (contrast_feature @ contrast_feature.T) / temperature
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()  # all < 0 # 因为对角线上是自己和自己，所以余弦相似度最大
        # tile mask
        mask = mask.repeat(c, c)
        logits_mask = 1 - torch.eye(mask.shape[0]).to(
            logits.device
        )  # == 1- torch.eye(mask.shape[0])
        mask = mask * logits_mask
        # compute log_prob

        exp_logits = torch.exp(logits) * logits_mask  # no identity
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 已经是论文中所求的结果
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(
            1
        )  # 分子都是一对样本具备相同语义且不是同一对样本/分母都是一对样本具备相同语义且不是同一对样本
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

    def Loglikeli(self, mu, logvar, y_samples):
        return (-((mu - y_samples) ** 2) / logvar.exp() - logvar).mean()

    def Club(self, mu, logvar, y_samples, t_mu, t_logvar, t_y_samples):

        # sample_size = y_samples.shape[0]
        # # random_index = torch.randint(sample_size, (sample_size,)).long()
        # random_index = torch.randperm(sample_size).long()
        #
        # positive = - (mu - y_samples) ** 2 / logvar.exp()
        # negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()  # log(z_j^+|z_i)
        # upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound / 2. # TODO; CLUB Sample

        positive = -((mu - y_samples) ** 2) / 2.0 / logvar.exp()
        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        student_mi = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return student_mi

    def Mmd(self, e1, e2, target, num_class):
        return conditional_mmd_rbf(e1, e2, target, num_class)

    def reset(self):
        self.begin_tloss = 0.0
        self.begin_ttop1 = 0.0
        self.begin_vloss = 0.0
        self.begin_vtop1 = 0.0
        self.best_acc = 0.0

    def run_one_train_batch_size(self, batch_idx, input, target):

        # TODO: turn target (N,1) -> (N,C)
        input = input.float().cuda()
        target = target.cuda()
        target = target.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).float()

        # TODO: Learning to diversify
        rand_choose = torch.randperm(input.shape[0])[: int(self.augmented_ratio * input.shape[0])]
        temp = input.clone()
        target_temp = target.clone()
        if rand_choose.shape[0] > 0:
            temp[rand_choose], target_temp[rand_choose] = self.augmentation(
                temp[rand_choose], target_temp[rand_choose]
            )
        inputs_max = self.convertor(temp)
        # inputs_max = inputs_max * 0.6 + input * 0.4
        # inputs_max=temp
        b, c, h, w = inputs_max.shape
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])
        if self.epoch >= self.yaml["warmup_epoch"]:
            adjust_lr(self.optimizer, self.epoch, self.yaml, batch_idx, len(self.dataloader))
        with torch.cuda.amp.autocast(enabled=True):
            (student_tuple, student_logits) = self.student_model(data_aug, is_feat=True)
            with torch.no_grad():
                (teacher_tuple, teacher_logits) = self.teacher_model(data_aug, is_feat=True)
            student_lambda = student_tuple[-1]
            teacher_lambda = teacher_tuple[-1]
            student_avgpool = self.avgpool2d(student_lambda)
            teacher_avgpool = self.avgpool2d(teacher_lambda)
            student_logvar = self.p_logvar(student_avgpool)
            student_mu = self.p_mu(student_avgpool)
            teacher_logvar = self.p_logvar(teacher_avgpool)
            teacher_mu = self.p_mu(teacher_avgpool)
            student_embedding = self.reparametrize(student_mu, student_logvar)
            teacher_embedding = self.reparametrize(teacher_mu, teacher_logvar)
        # TODO: compute relative loss

        # TODO: 1, vanilla KD Loss
        vanilla_kd_loss = self.KDLoss(student_logits.float(), teacher_logits.float(), labels)

        # TODO: 2. Lilikehood Loss student and teacher
        augmented_studnet_mu = student_mu[:b]
        augmented_student_logvar = student_logvar[:b]
        likeli_loss = -self.Loglikeli(
            augmented_studnet_mu, augmented_student_logvar, student_embedding[b:]
        )
        augmented_teacher_mu = teacher_mu[:b]
        augmented_teacher_logvar = teacher_logvar[:b]
        new_mu = torch.cat([augmented_studnet_mu, augmented_teacher_mu])
        new_logvar = torch.cat([augmented_student_logvar, augmented_teacher_logvar])
        new_embedding = torch.cat([student_embedding[b:], teacher_embedding[b:]])
        likeli_loss += -self.Loglikeli(new_mu, new_logvar, new_embedding)

        # TODO: 3. Combine all Loss in stage one
        loss_1 = self.weights[0] * vanilla_kd_loss + self.weights[1] * likeli_loss
        self.optimizer.zero_grad()
        self.scaler.scale(loss_1).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        original_sample = input[0]
        change_tensor_to_image(original_sample, "output", f"original{self.epoch}_sample")
        augment_sample = inputs_max[0]
        change_tensor_to_image(augment_sample, "output", f"augment{self.epoch}_sample")
        # TODO: Second Stage
        rand_choose = torch.randperm(input.shape[0])[: int(self.augmented_ratio * input.shape[0])]
        temp = input.clone()
        target_temp = target.clone()
        if rand_choose.shape[0] > 0:
            temp[rand_choose], target_temp[rand_choose] = self.augmentation(
                temp[rand_choose], target_temp[rand_choose]
            )
        inputs_max = self.convertor(temp, estimation=True)
        # inputs_max = inputs_max * 0.6 + input * 0.4
        # inputs_max=temp
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])
        b, c, h, w = inputs_max.shape

        with torch.cuda.amp.autocast(enabled=True):
            student_tuples, student_logits = self.student_model(data_aug, is_feat=True)
            teacher_tuples, teacher_logits = self.teacher_model(data_aug, is_feat=True)
            student_lambda = student_tuples[-1]
            teacher_lambda = teacher_tuples[-1]
            student_avgpool = self.avgpool2d(student_lambda)
            teacher_avgpool = self.avgpool2d(teacher_lambda)
            student_logvar = self.p_logvar(student_avgpool)
            student_mu = self.p_mu(student_avgpool)
            teacher_logvar = self.p_logvar(teacher_avgpool)
            teacher_mu = self.p_mu(teacher_avgpool)
            student_embedding = self.reparametrize(student_mu, student_logvar)
            teacher_embedding = self.reparametrize(teacher_mu, teacher_logvar)

        # TODO: 1. Club loss (互信息上界，减小增强样本与原始样本相关性)
        augmented_student_logvar = student_logvar[:b]
        augmented_student_mu = student_mu[:b]
        augmented_teacher_logvar = teacher_logvar[:b]
        augmented_teacher_mu = teacher_mu[:b]
        club_loss = self.Club(
            augmented_student_mu,
            augmented_student_logvar,
            student_embedding[b:],
            augmented_teacher_mu,
            augmented_teacher_logvar,
            teacher_embedding[b:],
        )

        # TODO: 3. Task Loss (确保他能够被正确识别，同时非正确类损失具备多样性。)
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()
        # TODO: 仅仅只有二分之一，因此需要扩张
        aug_logits, ori_logits = torch.chunk(student_logits, 2, 0)
        t_aug_logits, t_ori_logits = torch.chunk(teacher_logits, 2, 0)
        distance1 = (
            -F.kl_div(
                (aug_logits / self.yaml["criticion"]["temperature"]).log_softmax(1),
                (t_aug_logits / self.yaml["criticion"]["temperature"]).softmax(1),
                reduction="batchmean",
            )
            * (self.yaml["criticion"]["temperature"] ** 2)
        )
        distance2 = F.kl_div(t_aug_logits.log_softmax(1), target, reduction="batchmean")
        task_loss = distance2 * 0.1 + distance1 * 0.8 # left 0.8 right `.1

        # TODO: 4.to Combine all Loss in stage two
        loss_2 = (
            + self.weights[2] * club_loss
            + self.weights[3] * task_loss
        )

        # TODO: update params
        self.convertor_optimizer.zero_grad()
        self.scaler.scale(loss_2).backward()
        self.scaler.step(self.convertor_optimizer)
        self.scaler.update()

        # TODO: Compute top1 and top5
        top1, top5 = correct_num(student_logits[: input.shape[0]], target, topk=(1, 5))
        loss = loss_1 + loss_2
        lr = (
            self.scheduler.get_lr()[0] if hasattr(self.scheduler, "get_lr") else self.scheduler.lr()
        )
        self.log(self.teacher_model, loss.cpu(), top1.cpu(), top5.cpu(), lr)
        return (
            top1.cpu().item(),
            vanilla_kd_loss.cpu().item(),
            likeli_loss.cpu().item(),
            club_loss.cpu().item(),
            task_loss.cpu().item(),
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
        return top1.cpu().item(), loss.cpu().item()

    def run_one_train_epoch(self):
        """============================train=============================="""
        if self.epoch >= self.yaml["warmup_epoch"]:
            adjust_lr(self.optimizer, self.epoch, self.yaml)
        start_time = time.time()
        self.student_model.train()
        self.p_logvar.train()
        self.p_mu.train()
        self.convertor.train()
        self.log.train(len_dataset=len(self.dataloader))
        for batch_idx, (input, target) in enumerate(self.dataloader):
            (
                top1,
                vanilla_kd_loss,
                likeli_loss,
                club_loss,
                task_loss,
            ) = self.run_one_train_batch_size(batch_idx, input, target)
            self.wandb.log(
                {
                    "top1": top1,
                    "vanilla_kd_loss": vanilla_kd_loss,
                    "likeli_loss": likeli_loss,
                    "club_loss": club_loss,
                    "task_loss": task_loss,
                },
                step=self.accumuate_count,
            )
            self.accumuate_count += 1
        train_acc, train_loss = (
            self.log.epoch_state["top_1"] / self.log.epoch_state["steps"],
            self.log.epoch_state["loss"] / self.log.epoch_state["steps"],
        )
        use_time = round((time.time() - start_time) / 60, 2)
        self.ff.write(
            f"epoch:{self.epoch}, train_acc:{train_acc}, train_loss:{train_loss}, min:{use_time}\n"
        )
        return train_acc, train_loss, self.episode

    @torch.no_grad()
    def run_one_val_epoch(self):
        """============================val=============================="""
        start_time = time.time()
        self.student_model.eval()
        self.p_logvar.eval()
        self.p_mu.eval()
        self.convertor.eval()
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
            self.log(self.student_model, loss.cpu(), top1.cpu(), top5.cpu())
        test_acc, test_loss = (
            self.log.epoch_state["top_1"] / self.log.epoch_state["steps"],
            self.log.epoch_state["loss"] / self.log.epoch_state["steps"],
        )
        use_time = round((time.time() - start_time) / 60, 2)
        self.ff.write(
            f"epoch:{self.epoch}, test_acc:{test_acc}, test_loss:{test_loss}, min:{use_time}\n"
        )
        return test_acc

    def scheduler_step(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
            self.scheduler.step()
        elif isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
            self.scheduler.step(self.epoch)

        if isinstance(self.convertor_scheduler, torch.optim.lr_scheduler.MultiStepLR):
            self.convertor_scheduler.step()
        elif isinstance(self.convertor_scheduler, timm.scheduler.scheduler.Scheduler):
            self.convertor_scheduler.step(self.epoch)

    def training_in_all_epoch(self):
        for i in range(self.total_epoch):
            ttop1, tloss, _ = self.run_one_train_epoch()
            self.scheduler_step()
            vtop1 = self.run_one_val_epoch()
            self.wandb.log(
                {"train_loss": tloss, "train_top1": ttop1, "val_top1": vtop1}, step=self.epoch
            )
            self.epoch += 1
            if self.best_acc < vtop1:
                self.best_acc = vtop1
                path = self.model_save_path
                if not os.path.isdir(path):
                    os.makedirs(path)
                model_path = os.path.join(
                    self.model_save_path,
                    f"_epoch_{i}_dataset_{self.yaml['data']}_teacher_{self.yaml['tarch']}_student_{self.yaml['arch']}",
                )
                dict = {
                    "epoch": self.epoch,
                    "optimizer": self.optimizer.state_dict(),
                    "model": self.student_model.state_dict(),
                    "acc": vtop1,
                }
                torch.save(dict, model_path)
        self.ff.close()
        self.log.flush()
        self.wandb.finish()
