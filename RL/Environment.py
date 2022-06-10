import timm.scheduler.scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from datas.CIFAR100 import PolicyDatasetC100
from datas.CIFAR100 import SubPolicy
from datas.CIFAR10 import PolicyDatasetC10
from torch.utils.data import DataLoader
from helpers.adjust_lr import adjust_lr
from helpers.correct_num import correct_num
from helpers.log import Log
from RL.SACContinuous import SACContinuous
from RL.SAC import SAC
from RL.rl_utils import ReplayBuffer
import copy, time, math,random,os,sys
import numpy as np


def get_index(index, p, t):
    policy_list = [lambda x: SubPolicy(p, 'autocontrast', x),
                   lambda x: SubPolicy(p, 'contrast', x),
                   lambda x: SubPolicy(p, 'posterize', x),
                   lambda x: SubPolicy(p, 'solarize', x),
                   lambda x: SubPolicy(p, 'translateY', x),
                   lambda x: SubPolicy(p, 'shearX', x),
                   lambda x: SubPolicy(p, 'brightness', x),
                   lambda x: SubPolicy(p, 'shearY', x),
                   lambda x: SubPolicy(p, 'translateX', x),
                   lambda x: SubPolicy(p, 'sharpness', x),
                   lambda x: SubPolicy(p, 'invert', x),
                   lambda x: SubPolicy(p, 'color', x),
                   lambda x: SubPolicy(p, 'equalize', x),
                   lambda x: SubPolicy(p, 'rotate', x)]
    return policy_list[index](t)

def get_index_list(p):
    result=[]
    amplitude=[]
    len=14
    for i in range(len):
        randint=random.randint(0,9)
        result.append(get_index(i,p,randint))
        amplitude.append(randint)
    return result,amplitude

class PolicyEnv(object):
    def __init__(self, dataloader: DataLoader, valloader: DataLoader, testloader: DataLoader, student_model: nn.Module,
                 teacher_model: nn.Module, scheduler, optimizer: torch.optim.Optimizer, loss: nn.Module,
                 yaml,wandb,device=None):
        super(PolicyEnv, self).__init__()
        self.dataloader = dataloader
        self.valloader = valloader
        self.testloader = testloader
        assert isinstance(self.dataloader.dataset,torch.utils.data.Subset) and (isinstance(self.dataloader.dataset.dataset, PolicyDatasetC100) or isinstance(self.dataloader.dataset.dataset,PolicyDatasetC10))
        self.policies = copy.deepcopy(self.dataloader.dataset.dataset.policies)
        self.status=self.reset()
        if device!=None:
            self.student_model = student_model.to(device)
            self.teacher_model = teacher_model.to(device)
        else:
            self.student_model = student_model
            self.teacher_model = teacher_model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = 0
        self.total_epoch=yaml['epoch']
        self.yaml = yaml
        self.log = Log(log_each=yaml['log_each'])
        self.sample_num = yaml['sample_num']
        self.minimal_size=yaml['minimal_size']
        self.model_save_path=yaml['model_save_path']
        self.wandb=wandb
        self.reset_momentum()
        self.momentum = 0.999
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.done=False
        self.episode=0.
        self.SAC = SAC(
            state_dim=len(self.policies),
            hidden_dim=128,
            action1_dim=15,
            action2_dim=10,
            actor_lr=3e-4,
            critic_lr=3e-3,
            alpha_lr=3e-4,
            target_entropy=5,
            tau=0.001,
            gamma=0.999,
            device=self.device)
        self.buffer_size=yaml['buffer_size']
        self.replay_buffer=ReplayBuffer(self.buffer_size)
        self.rl_batch_size=yaml['rl_batch_size']
        self.scaler = torch.cuda.amp.GradScaler()
        time_path = time.strftime("%Y^%m^%d^%H^%M^%S", time.localtime()) + ".txt"
        self.ff = open(time_path, 'w')
    def reset_momentum(self):
        self.begin_tloss = 0.
        self.begin_ttop1 = 0.
        self.begin_vloss = 0.
        self.begin_vtop1 = 0.
        self.best_acc=0.
    def step(self, action1,action2, p=0.25):
        assert 0 <= action2 and action2 <= 9 and isinstance(action2, int)
        assert 0 <= action1 and action1 <= 14 and isinstance(action1,int)
        reward = 0
        if action1<14:
            self.dataloader.dataset.dataset.policies[action1]=get_index(action1,p,action2)
            self.status[action1] = action2
        else:
            reward=1
        return self.status,reward

    def reset(self,p=0.25):
        self.dataloader.dataset.dataset.policies,amplitude = get_index_list(p)
        return amplitude

    def run_one_train_batch_size(self, batch_idx, input, target):
        input = input.float().cuda()
        target = target.cuda()
        if input.ndim == 5:
            b, m, c, h, w = input.shape
            input = input.view(-1, c, h, w)
            target = target.view(-1)
        if self.epoch >= self.yaml['warmup_epoch']:
            lr = adjust_lr(self.optimizer, self.epoch, self.yaml, batch_idx, len(self.dataloader))
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            logits = self.student_model(input).float()
        with torch.no_grad():
            t_logits = self.teacher_model(input)
        loss = self.loss(logits, t_logits, target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        top1, top5 = correct_num(logits, target, topk=(1, 5))
        lr=self.scheduler.get_lr()[0] if hasattr(self.scheduler,"get_lr") else self.scheduler.lr()
        self.log(self.teacher_model, loss.cpu(), top1.cpu(), top5.cpu(), lr)
        return top1.cpu().item(),loss.cpu().item()
    def generate_val_sample(self,dataset):
        valdataset = dataset
        index = np.random.choice(len(valdataset), self.sample_num, replace=False)
        vinput,vtarget=[],[]
        for i in index:
            one_input,one_target=valdataset[i]
            vinput.append(one_input)
            vtarget.append(one_target)
        return torch.stack(vinput,0),torch.stack(vtarget,0)

    @torch.no_grad()
    def run_one_val_batch_size(self, input, target):
        input = input.float().cuda()
        target = target.cuda()
        if input.ndim == 5:
            b, m, c, h, w = input.shape
            input = input.view(-1, c, h, w)
            target = target.view(-1)
        logits = self.student_model(input).float()
        t_logits = self.teacher_model(input)
        loss = self.loss(logits, t_logits, target)
        top1, top5 = correct_num(logits, target, topk=(1, 5))
        return top1.cpu().item(), loss.cpu().item()
    def offline_sample_update(self,):
        if self.replay_buffer.size() > self.minimal_size:
            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.rl_batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            self.SAC.update(transition_dict)
    def run_one_train_epoch(self):
        """============================train=============================="""
        if self.epoch >= self.yaml['warmup_epoch']:
            lr = adjust_lr(self.optimizer, self.epoch, self.yaml)
        start_time = time.time()
        self.student_model.train()
        self.log.train(len_dataset=len(self.dataloader))
        self.episode=0
        for batch_idx, (input, target) in enumerate(self.dataloader):
            status=self.status # (14,)
            action1,action2=self.SAC.take_action(status)
            next_status, reward = self.step(action1,action2)
            top1,loss = self.run_one_train_batch_size(batch_idx, input, target)
            train_reward = -(loss - self.begin_tloss) + (top1 - self.begin_ttop1) / 10
            self.begin_tloss = self.momentum * loss + (1 - self.momentum) * self.begin_tloss if self.begin_tloss!=0 else loss
            self.begin_ttop1 = self.momentum * top1 + (1 - self.momentum) * self.begin_ttop1 if self.begin_ttop1!=0 else top1
            vinput, vtarget = self.generate_val_sample(self.valloader.dataset)
            top1, loss = self.run_one_val_batch_size(vinput, vtarget)
            val_reward = -(loss - self.begin_vloss) + (top1 - self.begin_vtop1) / 10
            self.begin_vloss = self.momentum * loss + (1 - self.momentum) * self.begin_vloss if self.begin_vloss!=0 else loss
            self.begin_vtop1 = self.momentum * top1 + (1 - self.momentum) * self.begin_vtop1 if self.begin_vtop1!=0 else top1
            self.wandb.log({'train_reward':train_reward,'val_reward':val_reward,'fix_reward':reward})
            reward = train_reward + val_reward + reward
            self.replay_buffer.add(status,[action1,action2],reward,next_status,done=False)
            self.episode+=reward
            self.offline_sample_update()
        train_acc, train_loss = self.log.epoch_state["top_1"] / self.log.epoch_state["steps"], self.log.epoch_state[
            "loss"] / self.log.epoch_state["steps"]
        use_time = round((time.time() - start_time) / 60, 2)
        self.ff.write(f"epoch:{self.epoch}, train_acc:{train_acc}, train_loss:{train_loss}, min:{use_time}\n")
        return train_acc,train_loss,self.episode

    @torch.no_grad()
    def run_one_val_epoch(self):
        """============================val=============================="""
        start_time = time.time()
        self.student_model.eval()
        self.log.eval(len_dataset=len(self.testloader))
        for batch_idx, (input, target) in enumerate(self.testloader):
            input = input.float().cuda()
            target=target.cuda()
            input.requires_grad_()
            torch.cuda.synchronize()
            logits = self.student_model(input)
            torch.cuda.synchronize()
            loss=F.cross_entropy(logits,target,reduction="mean")
            top1, top5 = correct_num(logits, target, topk=(1, 5))
            self.log(self.student_model, loss.cpu(),top1.cpu(), top5.cpu())
        test_acc, test_loss = self.log.epoch_state["top_1"] / self.log.epoch_state["steps"], self.log.epoch_state[
            "loss"] / self.log.epoch_state["steps"]
        use_time = round((time.time() - start_time) / 60, 2)
        self.ff.write(f"epoch:{self.epoch}, test_acc:{test_acc}, test_loss:{test_loss}, min:{use_time}\n")
        return test_acc
    def scheduler_step(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
            self.scheduler.step()
        elif isinstance(self.scheduler, timm.scheduler.scheduler.Scheduler):
            self.scheduler.step(self.epoch)
    def training_in_all_epoch(self):
        for i in range(self.total_epoch):
            ttop1,tloss,_=self.run_one_train_epoch()
            self.scheduler_step()
            vtop1=self.run_one_val_epoch()
            self.wandb.log({'train_loss':tloss,'train_top1':ttop1,'val_top1':vtop1},self.epoch)
            self.epoch+=1
            self.log.flush()
            if self.best_acc<vtop1:
                self.best_acc=vtop1
                path=self.model_save_path[0:-8]
                if not os.path.isdir(path):
                    os.makedirs(path)
                dict={
                    "epoch":self.epoch,
                    "optimizer":self.optimizer.state_dict(),
                    'model':self.student_model.state_dict(),
                    'acc':vtop1
                }
                torch.save(dict,self.model_save_path)
        self.ff.close()
        self.wandb.finish()




