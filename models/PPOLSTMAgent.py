import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from utils.sys import make_dir

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class CNNLSTMBlock(nn.Module):
    def __init__(self):
        super(CNNLSTMBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 16 * 16, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(input_size=32 * 2 * 2, hidden_size=200, batch_first=True)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        img_channel, img_height, img_width = inputs.shape[2], inputs.shape[3], inputs.shape[4]

        stacked_input = inputs.reshape(batch_size * seq_len, img_channel, img_height, img_width)
        x = self.conv_encoder(stacked_input)
        x = x.reshape(-1, 32 * 16 * 16)
        x = self.fc_layer(x)
        unstacked_input = x.reshape(batch_size, seq_len, 128)

        output, _ = self.lstm(unstacked_input)
        output = output.reshape(batch_size, -1)

        return output


class Actor(nn.Module):
    def __init__(self, action_dim, seq_len):
        super(Actor, self).__init__()

        self.feature_layer = CNNLSTMBlock()
        self.layers = nn.Sequential(
            nn.Linear(200 * seq_len, 100 * seq_len),
            nn.BatchNorm1d(100 * seq_len),
            nn.ReLU(inplace=True),
            nn.Linear(100 * seq_len, 4 * seq_len),
            nn.BatchNorm1d(4 * seq_len),
            nn.ReLU(inplace=True),
            nn.Linear(4 * seq_len, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.layers(x)
        return x


class Critic(nn.Module):
    def __init__(self, seq_len):
        super(Critic, self).__init__()

        self.feature_layer = CNNLSTMBlock()
        self.layers = nn.Sequential(
            nn.Linear(200 * seq_len, 100 * seq_len),
            nn.BatchNorm1d(100 * seq_len),
            nn.ReLU(inplace=True),
            nn.Linear(100 * seq_len, 4 * seq_len),
            nn.BatchNorm1d(4 * seq_len),
            nn.ReLU(inplace=True),
            nn.Linear(4 * seq_len, 1),
        )

    def forward(self, x):
        x = self.feature_layer(x)
        x = self.layers(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, action_dim, lstm_length, device, action_std_init=0.6):
        super(ActorCritic, self).__init__()

        self.device = device

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.actor = Actor(action_dim, seq_len=lstm_length)
        self.critic = Critic(seq_len=lstm_length)

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPOLSTMAgent(nn.Module):

    def __init__(self,
                 name,
                 lr_actor,
                 lr_critic,
                 lstm_length,
                 obs_dim,
                 action_dim,
                 batch_size,
                 max_grad_norm, gamma, eps_clip, K_epochs, entropy_coef, v_loss_coef,
                 device
                 ):
        super(PPOLSTMAgent, self).__init__()

        self.name = name
        self.batch_size = batch_size
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.v_loss_coef = v_loss_coef

        self.buffer = RolloutBuffer()

        self.lstm_length = lstm_length

        self.policy = ActorCritic(action_dim, lstm_length, device=device).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(action_dim, lstm_length, device=device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.to(device)

    def act(self, obs):
        self.eval()

        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        self.train()
        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
