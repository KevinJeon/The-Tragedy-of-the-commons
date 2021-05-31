import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils.sys import make_dir


class CNNLSTMBlock(nn.Module):
    def __init__(self):
        super(CNNLSTMBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTMCell(32 * 2 * 2, 200)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = self.conv_encoder(inputs)
        x = x.view(-1, 32 * 2 * 2)
        hx, cx = self.lstm(x, (hx, cx))

        return hx, cx


class A2CLSTMAgent(nn.Module):

    def __init__(self,
                 name,
                 lr,
                 eps,
                 alpha,
                 obs_dim,
                 action_dim,
                 batch_size,
                 device
                 ):
        super(A2CLSTMAgent, self).__init__()

        self.name = name
        self.batch_size = batch_size
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.alpha = alpha

        self.encoder = CNNLSTMBlock()
        self.lstm = nn.LSTMCell(32 * 2 * 2, self.lstm_size)

        self.actor = nn.Linear(200, 1)
        self.critic = nn.Linear(200, action_dim)


    def act(self, obs, sample=False) -> float:
        self.encoder(obs)


        pass

    def train(self, batch):
        pass

    def save(self, num):
        make_dir('models')
        make_dir(f'models/{num}')

        torch.save(self.state_dict(), os.path.join('.', 'models', str(num), f'{self.name}.pth'))


