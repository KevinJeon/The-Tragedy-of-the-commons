import torch as tr

class RolloutStorage(object):

    def __init__(self, agent_type, num_agent, num_step, batch_size, num_obs, num_action, num_rec):
        self.obs = tr.zeros(num_agent, num_step + 1, *num_obs)
        self.rew = tr.zeros(num_agent, num_step, 1)
        self.ret = tr.zeros(num_agent, num_step + 1, 1)
        self.act = tr.zeros(num_agent, num_step, num_action)
        self.mask = tr.zeros(num_agent, num_step + 1, 1) # If mask == 0, Terminal
        self.num_step = num_step
        self.step = 0
        self.num_agent = num_agent
        self.agent_type = agent_type
        
        self.onehot = tr.eye(num_action)
        # For CPC
        if agent_type == 'ac':
            self.s_feat = tr.zeros(num_agent, num_step, num_action)
            self.a_feat = tr.zeros(num_agent, num_step, 128)
            self.h = tr.zeros(num_agent, num_step + 1, num_rec)
            self.val = tr.zeros(num_agent, num_step + 1, 1)
            self.logprob = tr.zeros(num_agent, num_step, 1)

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
        obss = tr.from_numpy(obss)
        rews = tr.tensor(rews)
        masks = tr.tensor(masks)
        for i in range(self.num_agent):
            self.obs[i, self.step + 1].copy_(obss[i])
            self.act[i, self.step].copy_(self.onehot[acts[i]].view(-1))
            self.rew[i, self.step].copy_(rews[i])
            self.mask[i, self.step + 1].copy_(masks[i])
            
            # For CPC
            if self.agent_type == 'ac':
                v, logprob, h, s_f, a_f = infos[i]
                self.h[i, self.step + 1].copy_(h.view(-1))
                self.logprob[i, self.step].copy_(logprob.view(-1))
                self.val[i, self.step].copy_(v.view(-1))
                self.s_feat[i, self.step].copy_(s_f.view(-1))
                self.a_feat[i, self.step].copy_(a_f.view(-1))
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
            

