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

import wandb
from datas.LearningAutoAugment import LearningAutoAugment
from helpers.correct_num import correct_num
from helpers.log import Log

# from losses.ReviewKD import ReviewKD
# from losses.PRviewKD import ReviewKD
from losses.DFD import DynamicFeatureDistillation
from losses.SimpleMseKD import SMSEKD
from utils.mmd import conditional_mmd_rbf


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
        if yaml["LAA"]["augmentation_policy"] == "cifar10":
            policy_type = AutoAugmentPolicy.CIFAR10
        elif yaml["LAA"]["augmentation_policy"] == "imagenet":
            policy_type = AutoAugmentPolicy.IMAGENET
        else:
            raise NotImplementedError
        print(policy_type)
        self.convertor = (
            DDP(
                LearningAutoAugment(
                    policy=policy_type,
                    C=yaml["LAA"]["C"],
                    H=yaml["LAA"]["H"],
                    W=yaml["LAA"]["W"],
                    p=yaml["LAA"]["p"],
                    num_train_samples=len(self.dataloader.dataset),
                    total_epoch=self.total_epoch,
                ).cuda(gpu),
                device_ids=[gpu],
            )
            if torch.cuda.is_available()
            else LearningAutoAugment(
                policy=policy_type,
                C=yaml["LAA"]["C"],
                H=yaml["LAA"]["H"],
                W=yaml["LAA"]["W"],
                p=yaml["LAA"]["p"],
                num_train_samples=len(self.dataloader.dataset),
                total_epoch=self.total_epoch,
            ).cpu()
        )

        self.convertor_optimizer = torch.optim.SGD(
            self.convertor.parameters(),
            lr=yaml["sc_lr"],
            momentum=0.9,
            nesterov=True,
        )
        if yaml['online'] == True:
            self.convertor_optimizer.add_param_group({"params": self.teacher_model.parameters()})
        else:
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        self.convertor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.convertor_optimizer,
            T_max=self.yaml["epoch"],
            eta_min=1e-5,
        )
        self.tran = (
            transforms.Compose(
                [transforms.Normalize([0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
            )
            if yaml["LAA"]["augmentation_policy"] == "cifar10"
            else transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        self.avgpool2d = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        if self.student_model.module.last_channel != self.teacher_model.module.last_channel:
            self.teacher_expand = DDP(
                nn.Linear(
                    self.teacher_model.module.last_channel, self.student_model.module.last_channel
                ).cuda(gpu),
                device_ids=[gpu],
            )
        else:
            self.teacher_expand = nn.Identity()

        self.p_logvar = (
            DDP(
                nn.Sequential(
                    nn.Linear(self.student_model.module.last_channel, 512),
                    nn.GELU(),
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.LeakyReLU(),
                ).cuda(gpu),
                device_ids=[gpu],
            )
            if torch.cuda.is_available()
            else nn.Sequential(
                nn.Linear(self.student_model.module.last_channel, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
            )
        )

        self.p_mu = (
            DDP(
                nn.Sequential(
                    nn.Linear(self.student_model.module.last_channel, 512),
                    nn.GELU(),
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.LeakyReLU(),
                ).cuda(gpu),
                device_ids=[gpu],
            )
            if torch.cuda.is_available()
            else nn.Sequential(
                nn.Linear(self.student_model.module.last_channel, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
            )
        )
        # TODO: It is important to remember to add the last parameter in the optimizer
        self.optimizer.add_param_group({"params": self.avgpool2d.parameters()})
        self.optimizer.add_param_group({"params": self.p_logvar.parameters()})
        self.optimizer.add_param_group({"params": self.p_mu.parameters()})
        self.optimizer.add_param_group({"params": self.teacher_expand.parameters()})
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
        self.freeze_modeules_list = [
            self.dfd,
            self.student_model,
            self.teacher_expand,
            self.p_mu,
            self.p_logvar,
        ]

        if yaml["resume"] != "none":
            dict = torch.load(yaml["resume"])
            self.optimizer.load_state_dict(dict["optimizer"])
            self.scheduler.load_state_dict(dict["scheduler"])
            self.student_model.load_state_dict(dict["student_model"])
            self.p_mu.load_state_dict(dict["p_mu"])
            self.p_logvar.load_state_dict(dict["p_logvar"])
            self.convertor.load_state_dict(dict["convertor"], strict=False)
            self.convertor_optimizer.load_state_dict(dict["convertor_optimizer"])
            self.convertor_scheduler.load_state_dict(dict["convertor_scheduler"])
            self.scaler.load_state_dict(dict["scaler"])
            self.best_acc = dict["acc"]
            self.begin_epoch = self.epoch = dict["epoch"] + 1
            print(f"successfully load checkpoint from {yaml['resume']}")

    def reset_parameters(self, modules):
        for module in modules.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight.data, 0, 0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def _freeze_parameters(self, modeules):
        for module in modeules:
            for i in module.modules():
                i.requires_grad = False

    def _unfreeze_parameters(self, modeules):
        for module in modeules:
            for i in module.modules():
                i.requires_grad = False

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
        return hard_loss + (temperature ** 2) * soft_loss * self.yaml['criticion']['alpha']

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

    def Club(self, mu, logvar, y_samples):

        positive = -((mu - y_samples) ** 2) / 2.0 / logvar.exp()
        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        student_mi = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return student_mi

    def Mmd(self, e1, e2, target, num_class):
        return conditional_mmd_rbf(e1, e2, target, num_class)

    def run_one_train_batch_size(self, batch_idx, indexs, input, target):
        # TODO: turn target (N,1) -> (N,C)
        input = input.float().cuda(self.gpu)
        target = target.cuda(self.gpu)
        target = target.view(-1)
        target = F.one_hot(target, num_classes=self.num_classes).float()
        self._unfreeze_parameters(self.freeze_modeules_list)
        # TODO: Learning to diversify
        with torch.no_grad():
            inputs_max, target_temp, attention_index = self.convertor.module(
                input.clone(), target, indexs, 2 * self.epoch
            )
        b, c, h, w = inputs_max.shape
        data_aug = torch.cat([inputs_max, input])
        labels = torch.cat([target_temp, target])

        with torch.cuda.amp.autocast(enabled=True):
            (student_tuple, student_logits) = self.student_model(data_aug, is_feat=True)
            with torch.no_grad():
                (teacher_tuple, teacher_logits) = self.teacher_model(data_aug, is_feat=True)
            student_lambda = student_tuple[-1]
            student_tuple = student_tuple[:-1]
            teacher_tuple = teacher_tuple[:-1]
            student_avgpool = self.avgpool2d(student_lambda)
            student_logvar = self.p_logvar(student_avgpool)
            student_mu = self.p_mu(student_avgpool)
            student_embedding = self.reparametrize(student_mu, student_logvar)
        # TODO: compute relative loss

        # TODO: 1, vanilla KD Loss
        vanilla_kd_loss = self.KDLoss(
            student_logits.float(),
            teacher_logits.float(),
            labels,
            self.yaml["criticion"]["temperature"],
        )

        # TODO: 2. Lilikehood Loss student and teacher2
        augmented_studnet_mu = student_mu[:b]
        augmented_student_logvar = student_logvar[:b]
        if self.weights[1] == 0:
            likeli_loss = torch.Tensor([0.0]).cuda()
        else:
            likeli_loss = -self.Loglikeli(
                augmented_studnet_mu, augmented_student_logvar, student_embedding[b:]
            )
        # TODO: 3 DFD Loss
        if self.weights[2] == 0:
            dfd_loss = torch.Tensor([0.0]).cuda(self.gpu)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                dfd_loss = self.dfd(student_tuple, teacher_tuple,labels,only_alignment=True)

        # TODO: 3. Combine all Loss in stage one
        loss_1 = (
            self.weights[0] * vanilla_kd_loss
            + self.weights[1] * likeli_loss
            + self.weights[2] * dfd_loss
        )
        # self.convertor_optimizer.zero_grad()
        self.optimizer.zero_grad()
        self.scaler.scale(loss_1).backward()
        nn.utils.clip_grad_norm_(self.dfd.parameters(), max_norm=2, norm_type=2)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # TODO: Second Stage

        if not self.only_satge_one:
            self._freeze_parameters(self.freeze_modeules_list)
            inputs_max, target_temp, _ = self.convertor(
                input.clone(), target, indexs, 2 * self.epoch + 1
            )
            data_aug = torch.cat([inputs_max, input])
            b, c, h, w = inputs_max.shape
            with torch.cuda.amp.autocast(enabled=True):
                student_tuples, student_logits = self.student_model.module(data_aug, is_feat=True)
                teacher_tuples, teacher_logits = self.teacher_model.module(data_aug, is_feat=True)
                student_lambda = student_tuples[-1]
                student_tuples = student_tuples[:-1]
                teacher_tuples = teacher_tuples[:-1]
                student_avgpool = self.avgpool2d(student_lambda)
                student_logvar = self.p_logvar.module(student_avgpool)
                student_mu = self.p_mu.module(student_avgpool)
                student_embedding = self.reparametrize(student_mu, student_logvar)

            # TODO: 1. Club loss (互信息上界，减小增强样本与原始样本相关性)
            augmented_student_logvar = student_logvar[:b]
            augmented_student_mu = student_mu[:b]
            club_loss = self.Club(
                augmented_student_mu,
                augmented_student_logvar,
                student_embedding[b:],
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
            task_loss = distance2 * 0.1 + distance1 * 0.8  # left 0.8 right `.1

            # TODO: 4 negative dfd loss
            if self.weights[3] == 0:
                ne_dfd_loss = -torch.Tensor([0.0]).cuda(self.gpu)
            else:
                with torch.cuda.amp.autocast(enabled=True):
                    ne_dfd_loss = (
                        -self.dfd.module(student_tuples, teacher_tuples,labels,only_alignment=False)
                        * 0.8
                    )
            # TODO: 5.to Combine all Loss in stage two
            loss_2 = (
                +self.weights[3] * ne_dfd_loss
                + self.weights[4] * club_loss
                + self.weights[5] * task_loss
            )

            # TODO: update params
            self.convertor_optimizer.zero_grad()
            # self.optimizer.zero_grad()
            self.scaler.scale(loss_2).backward()
            nn.utils.clip_grad_norm_(self.dfd.parameters(), max_norm=2, norm_type=2)
            self.scaler.step(self.convertor_optimizer)
            self.scaler.update()
        else:
            ne_dfd_loss = torch.Tensor([0.0]).cuda(self.gpu)
            club_loss = torch.Tensor([0.0]).cuda(self.gpu)
            task_loss = torch.Tensor([0.0]).cuda(self.gpu)
            loss_2 = (
                +self.weights[3] * ne_dfd_loss
                + self.weights[4] * club_loss
                + self.weights[5] * task_loss
            )

        # TODO: Compute top1 and top5
        top1, top5 = correct_num(student_logits[: input.shape[0]], target, topk=(1, 5))
        dist.all_reduce(top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(top5, op=dist.ReduceOp.SUM)
        top1 /= torch.cuda.device_count()
        top5 /= torch.cuda.device_count()

        loss = loss_1 + loss_2
        lr = (
            self.scheduler.get_last_lr()
            if hasattr(self.scheduler, "get_lr")
            else self.scheduler.lr()
        )
        lr = lr[0]
        if self.gpu == 0:
            self.log(self.teacher_model, loss.cpu(), top1.cpu(), top5.cpu(), lr)
        return (
            top1.cpu().item(),
            vanilla_kd_loss.cpu().item(),
            dfd_loss.cpu().item(),
            ne_dfd_loss.cpu().item(),
            likeli_loss.cpu().item(),
            club_loss.cpu().item(),
            task_loss.cpu().item(),
            attention_index,
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
        self.p_logvar.train()
        self.p_mu.train()
        self.convertor.train()
        if self.gpu == 0:
            self.log.train(len_dataset=len(self.dataloader))
        for batch_idx, (index, input, target) in enumerate(self.dataloader):
            (
                top1,
                vanilla_kd_loss,
                dfd_loss,
                ne_dfd_loss,
                likeli_loss,
                club_loss,
                task_loss,
                attention_index,
            ) = self.run_one_train_batch_size(batch_idx, index, input, target)
            if self.gpu == 0:
                if batch_idx == 0:
                    self.ff.write(str(attention_index.cpu().numpy().tolist()) + "\n")
                self.wandb.log(
                    {
                        "top1": top1,
                        "vanilla_kd_loss": vanilla_kd_loss,
                        "dfd_loss": dfd_loss,
                        "ne_dfd_loss": ne_dfd_loss,
                        "likeli_loss": likeli_loss,
                        "club_loss": club_loss,
                        "task_loss": task_loss,
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
        self.p_logvar.eval()
        self.p_mu.eval()
        self.convertor.eval()
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
        if isinstance(self.convertor_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            self.convertor_scheduler.step()
        elif isinstance(self.convertor_scheduler, timm.scheduler.scheduler.Scheduler):
            self.convertor_scheduler.step(self.epoch)

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
                    "p_mu": self.p_mu.state_dict(),
                    "p_logvar": self.p_logvar.state_dict(),
                    "dfd": self.dfd.state_dict(),
                    "convertor": self.convertor.state_dict(),
                    "convertor_optimizer": self.convertor_optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "convertor_scheduler": self.convertor_scheduler.state_dict(),
                }
                torch.save(dict, model_path)

        if self.gpu == 0:
            self.log.flush()
            self.ff.close()
            self.wandb.finish()
