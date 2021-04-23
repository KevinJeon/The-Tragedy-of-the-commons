import torch as tr

class RolloutStorage(object):

    def __init__(self, num_step, batch_size, num_obs, num_action, num_rec):
        self.obs = tr.zeros(num_step + 1, *num_obs)
        self.h = tr.zeros(num_step + 1, num_rec)
        self.rew = tr.zeros(num_step, 1)
        self.val = tr.zeros(num_step + 1, 1)
        self.ret = tr.zeros(num_step + 1, 1)
        self.logprob = tr.zeros(num_step, 1)
        self.act = tr.zeros(num_step, num_action)
        self.mask = tr.zeros(num_step + 1, 1) # If mask == 0, Terminal
        self.num_step = num_step
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.h = self.h.to(device)
        self.rew = self.rew.to(device)
        self.val = self.val.to(device)
        self.ret = self.ret.to(device)
        self.logprob = self.logprob.to(device)
        self.act = self.act.to(device)
        self.mask = self.mask.to(device)

    def add(self, obs, h, act, logprob, v, rew, mask):
        self.obs[self.step + 1].copy_(obs)
        self.h[self.step + 1].copy_(h)
        self.act[self.step].copy_(act)
        self.logprob[self.step].copy_(logprob)
        self.val[self.step].copy_(v)
        self.rew[self.step].copy_(rew)
        self.mask[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_step

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.h[0].copy_(self.h[-1])
        self.mask[0].copy_(self.mask[0])
    
    def compute_return(self, v_next, gamma):
        self.ret[-1] = next_val
        for step in reversed(range(self.rew.size(0))):
            self.ret[step] = self.ret[step + 1] * gamma * self.mask[step + 1] + self.rew[step]
            

