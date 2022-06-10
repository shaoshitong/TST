import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import RL.rl_utils
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action1_dim,action2_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action1_dim)
        self.fc3 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim,action2_dim)


    def forward(self, x):
        action1 = self.fc2(F.relu(self.fc1(x))).softmax(1)
        action2 = self.fc4(F.relu(self.fc3(x))).softmax(1)
        return action1,action2


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action1_dim,action2_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action1_dim)
        self.fc3 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim,action2_dim)



    def forward(self, x):
        value1 = self.fc2(F.relu(self.fc1(x)))
        value2 = self.fc4(F.relu(self.fc3(x)))
        return value1,value2

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action1_dim,action2_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action1_dim,action2_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action1_dim,action2_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action1_dim,action2_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,action1_dim, action2_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,action1_dim, action2_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        if isinstance(state,np.ndarray):
            state=torch.from_numpy(state).float().to(self.device)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs1,probs2 = self.actor(state)
        action_dist1 = torch.distributions.Categorical(probs1)
        action1 = action_dist1.sample()
        action_dist2 = torch.distributions.Categorical(probs2)
        action2 = action_dist2.sample()
        return action1.item(),action2.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs1,next_probs2 = self.actor(next_states)
        next_probs=torch.cat([next_probs1,next_probs2],1)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value_1,q1_value_2 = self.target_critic_1(next_states)
        q1_value=torch.cat([q1_value_1,q1_value_2],1)
        q2_value_1,q2_value_2 = self.target_critic_2(next_states)
        q2_value=torch.cat([q2_value_1,q2_value_2],1)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions1 = torch.tensor(transition_dict['actions'][:,0]).view(-1, 1).to(
            self.device).long()  # 动作不再是float类型
        actions2 = torch.tensor(transition_dict['actions'][:,1]).view(-1,1).to(self.device).long()
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).sum(0,keepdim=True).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device).float()

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        q_values=self.critic_1(states)
        critic_1_q_values_1 = q_values[0].gather(1, actions1)
        critic_1_q_values_2 = q_values[1].gather(1, actions2)
        critic_1_q_values=critic_1_q_values_1+critic_1_q_values_2
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        q_values=self.critic_2(states)
        critic_2_q_values_1 = q_values[0].gather(1, actions1)
        critic_2_q_values_2 = q_values[1].gather(1, actions2)
        critic_2_q_values = critic_2_q_values_1 + critic_2_q_values_2
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs1,probs2 = self.actor(states)
        probs=torch.cat([probs1,probs2],1)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = torch.cat(self.critic_1(states),1)
        q2_value = torch.cat(self.critic_2(states),1)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)