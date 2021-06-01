import numpy as np
import torch


class ManagerReplayStorage(object):

    def __init__(self, capacity, num_step, batch_size, device):
        self.obs = np.zeros((capacity, num_step, 3, 256, 256))
        self.next_obs = np.zeros((capacity, num_step, 3, 256, 256))
        self.reward = np.zeros((capacity, 1))
        self.action = np.zeros((capacity, 1))

        self.capacity = capacity
        self.idx = 0
        self.device = device
        self.batch_size = batch_size

    def add(self, obs, action, reward, next_obs):
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_obs[self.idx] = next_obs

        self.idx = (self.idx + 1) % self.capacity

    def sample(self):
        ind = np.random.randint(0, len(self), size=self.batch_size)

        obses = torch.Tensor(self.obs[ind]).to(self.device)
        actions = torch.Tensor(self.action[self.idx]).to(self.device)
        rewards = torch.Tensor(self.reward[self.idx]).to(self.device)
        next_obses = torch.Tensor(self.next_obs[self.idx]).to(self.device)

        return obses, actions, rewards, next_obses

    def __len__(self):
        return self.idx if self.idx < self.capacity else self.capacity

