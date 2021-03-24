import numpy as np
import torch as tr

class OffpolicyMemory(object):
    def __init__(self, num_agent, obs_sizes, num_actions, max_step):
        self.max_step = max_step
        self.num_agent = num_agent
        self.obss = [[] for _ in range(num_agent)]
        self.acts = [[] for _ in range(num_agent)]
        self.rews = [[] for _ in range(num_agent)] 
        self.obss_next = [[] for _ in range(num_agent)]
        self.masks = [[] for _ in range(num_agent)]
    
    def add(self, ind, obs, act, rew, obs_next, mask):
        self.obss[ind].append(obs)
        self.acts[ind].append(act)
        self.rews[ind].append(rew)
        self.obss_next[ind].append(obs_next)
        self.masks[ind].append(mask)

    def sample(self, batch_size):
        inds = np.random.choice(np.arange(len(self.obss)), batch_size, replace=False)
        if self.use_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x : Variable(Tensor(x), reguires_grad=False)

        obss = [cast(np.array(self.obss[a]))[inds] for a in range(self.num_agent)]
        acts = [cast(np.array(self.acts[a]))[inds] for a in range(self.num_agent)]
        rews = [cast(np.array(self.rews[a]))[inds] for a in range(self.num_agent)]
        obss_next = [cast(np.array(self.obss_next[a]))[inds] for a in range(self.num_agent)]
        masks = [cast(np.array(self.masks[a]))[inds] for a in range(self.num_agent)]
        return obss, acts, rews, obss_next, masks
