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


class LearnDiversifyEnv(object):
    def __init__(
            self,
            dataloader: DataLoader,
            testloader: DataLoader,
            student_model: nn.Module,
            teacher_model: nn.Module,
            scheduler,
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
        AugNet = BigImageAugNet if not yaml['augnettype'] == "SmallImageAugNet" else SmallImageAugNet
        self.convertor = (
            AugNet(img_size=yaml["img_size"]).cuda()
            if torch.cuda.is_available()
            else AugNet(img_size=yaml["img_size"]).cpu()
        )
        self.convertor_optimizer = torch.optim.SGD(
            self.convertor.parameters(), lr=yaml["sc_lr"], momentum=0.9
        )
        self.tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.avgpool2d = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.p_logvar = (
            nn.Sequential(nn.Linear(self.student_model.last_channel, 512), nn.ReLU()).cuda()
            if torch.cuda.is_available()
            else nn.Sequential(nn.Linear(self.student_model.last_channel, 512), nn.ReLU())
        )
        self.p_mu = (
            nn.Sequential(nn.Linear(self.student_model.last_channel, 512), nn.LeakyReLU()).cuda()
            if torch.cuda.is_available()
            else nn.Sequential(nn.Linear(self.student_model.last_channel, 512), nn.LeakyReLU())
        )

        # TODO: It is important to remember to add the last parameter in the optimizer
        self.optimizer.add_param_group({"params": self.avgpool2d.parameters()})
        self.optimizer.add_param_group({"params": self.p_logvar.parameters()})
        self.optimizer.add_param_group({"params": self.p_mu.parameters()})
        self.weights = yaml["weights"]

        # TODO: Natural Aumentation (i.e., AutoAugment, Cutmix, YOCO)
        self.augmentation = Augmentation(num_classes=yaml['num_classes'], policy=yaml['augmentation_policy'],
                                         mode=yaml['augmentation_mode'])
        self.augmented_ratio = yaml['augmented_ratio']

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
        hard_loss = F.kl_div(torch.log_softmax(student_output, dim=-1), targets,
                             reduction='batchmean') if targets != None else 0.0
        return hard_loss + (temperature ** 2) * soft_loss

    def Contextual(self, a, b):
        acosineb = 1 - F.cosine_similarity(a, b, 1, 1e-8)  # (b,)
        distance = (a.pow(2).sum(-1).sqrt() - b.pow(2).sum(-1).sqrt()) + acosineb
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

    def Loglikeli(self, mu1, logvar1, mu2, logvar2):
        return (
                       -((mu1 - mu2) ** 2) / logvar2.exp() - logvar1.exp() / logvar2.exp() + logvar1 - logvar2
               ).mean() / 2

    def Club(self, o_mu, o_logvar, a_mu,
             a_logvar):  # TODO: mu,logvar -> augmentad sample , y_samples -> original sample
        sample_size = o_mu.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = -(-((
                               a_mu - o_mu) ** 2) / o_logvar.exp() - a_logvar.exp() / o_logvar.exp() + a_logvar / 2 - o_logvar / 2).mean()
        negative = (-((a_mu[random_index] - o_mu) ** 2) / o_logvar.exp() - a_logvar[random_index].exp() / o_logvar.exp()
                    + a_logvar[random_index] / 2 - o_logvar / 2).mean()
        upper_bound = positive + negative
        return upper_bound

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
        rand_choose = torch.randperm(input.shape[0])[:int(self.augmented_ratio * input.shape[0])]
        temp = input.clone()
        target_temp = target.clone()
        temp[rand_choose], target_temp[rand_choose] = self.augmentation(temp[rand_choose], target_temp[rand_choose])
        inputs_max = self.tran(torch.sigmoid(self.convertor(temp)))
        inputs_max = inputs_max * 0.6 + input * 0.4
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

        # TODO: 2. Maximize MI between original sample and augmented sample
        student_embedding_augmented = F.normalize(student_embedding[:b]).unsqueeze(1)
        student_embedding_original = F.normalize(student_embedding[b:]).unsqueeze(1)
        con_sup_loss = self.ConLoss(
            torch.cat([student_embedding_original, student_embedding_augmented], dim=1), target.argmax(1)
        )

        # TODO: 3. Lilikehood Loss
        original_studnet_mu = student_mu[b:]
        original_student_logvar = student_logvar[b:]
        augmented_studnet_mu = student_mu[:b]
        augmented_student_logvar = student_logvar[:b]
        likeli_loss = -self.Loglikeli(augmented_studnet_mu, augmented_student_logvar, original_studnet_mu,
                                      original_student_logvar)

        # TODO: 4. Combine all Loss in stage one
        loss_1 = (
                self.weights[0] * vanilla_kd_loss
                + self.weights[1] * likeli_loss
                + self.weights[2] * con_sup_loss
        )
        self.optimizer.zero_grad()
        self.scaler.scale(loss_1).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        original_sample = input[0]
        change_tensor_to_image(original_sample, "output", f"original{self.epoch}_sample")

        # TODO: Second Stage
        rand_choose = torch.randperm(input.shape[0])[:int(self.augmented_ratio * input.shape[0])]
        temp = input.clone()
        target_temp = target.clone()
        temp[rand_choose], target_temp[rand_choose] = self.augmentation(temp[rand_choose], target_temp[rand_choose])
        inputs_max = self.tran(torch.sigmoid(self.convertor(temp, estimation=True)))
        inputs_max = inputs_max * 0.6 + input * 0.4
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])
        b, c, h, w = inputs_max.shape
        augment_sample = inputs_max[0]
        change_tensor_to_image(augment_sample, "output", f"augment{self.epoch}_sample")

        with torch.cuda.amp.autocast(enabled=True):
            student_tuples, student_logits = self.student_model(data_aug, is_feat=True)
            student_lambda = student_tuples[-1]
            student_avgpool = self.avgpool2d(student_lambda)
            student_logvar = self.p_logvar(student_avgpool)
            student_mu = self.p_mu(student_avgpool)
            # student_embedding = self.reparametrize(student_mu, student_logvar)

        # TODO: 1. Club loss
        augmented_student_logvar = student_logvar[:b]
        original_student_logvar = student_logvar[b:]
        augmented_student_mu = student_mu[:b]
        original_student_mu = student_mu[b:]
        club_loss = self.Club(original_student_mu, original_student_logvar, augmented_student_mu,
                              augmented_student_logvar)

        # TODO: 2. Task Loss (确保他能够被正确识别，同时非正确类损失具备多样性。)
        student_logits=student_logits.float()
        real_mask = target.bool().repeat(2, 1)
        fake_mask = ~real_mask  # TODO: 仅仅只有二分之一，因此需要扩张
        tmp=student_logits[fake_mask]
        tmp=tmp.view(fake_mask.shape[0],-1)
        aug_logits, ori_logits = torch.chunk(tmp, 2, 0)
        contextual_loss = self.Contextual(aug_logits, ori_logits)
        cross_loss = -torch.log(student_logits[real_mask]).mean()
        task_loss = cross_loss + contextual_loss
        # TODO: 3. Combine all Loss in stage two
        loss_2 = self.weights[3] * club_loss + self.weights[4] * task_loss
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
            con_sup_loss.cpu().item(),
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
                con_sup_loss,
                club_loss,
                task_loss,
            ) = self.run_one_train_batch_size(batch_idx, input, target)
            self.wandb.log(
                {
                    "top1": top1,
                    "vanilla_kd_loss": vanilla_kd_loss,
                    "likeli_loss": likeli_loss,
                    "con_sup_loss": con_sup_loss,
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
