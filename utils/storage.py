import torch as tr

class RolloutStorage(object):

    def __init__(self, num_agent, num_step, batch_size, num_obs, num_action, num_rec):
        self.obs = tr.zeros(num_agent, num_step + 1, *num_obs)
        self.h = tr.zeros(num_agent, num_step + 1, num_rec)
        self.rew = tr.zeros(num_agent, num_step, 1)
        self.val = tr.zeros(num_agent, num_step + 1, 1)
        self.ret = tr.zeros(num_agent, num_step + 1, 1)
        self.logprob = tr.zeros(num_agent, num_step, 1)
        self.act = tr.zeros(num_agent, num_step, num_action)
        self.mask = tr.zeros(num_agent, num_step + 1, 1) # If mask == 0, Terminal
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

    def add(self, obss, acts, rews, masks, infos):
        for i, (obs, act, rew, mask, info) in enumerate(zip(obss, acts, rews, masks, infos)):
            self.obs[i, self.step + 1].copy_(obs)
            self.h[i, self.step + 1].copy_(h)
            self.act[i, self.step].copy_(act)
            self.logprob[i, self.step].copy_(logprob)
            self.val[i, self.step].copy_(v)
            self.rew[i, self.step].copy_(rew)
            self.mask[i, self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_step

    def after_update(self):
        for i in range(self.num_agent):
            self.obs[i, 0].copy_(self.obs[i, -1])
            self.h[i, 0].copy_(self.h[i, -1])
            self.mask[i, 0].copy_(self.mask[i, 0])
    
    def compute_return(self, v_nexts, gamma):
        for i, v_next in enumerate(v_nexts):
            self.ret[i, -1] = v_next 
            for step in reversed(range(self.rew.size(1))):
                self.ret[i, step] = self.ret[i, step + 1] * gamma * self.mask[i, step + 1] + self.rew[i, step]
            

