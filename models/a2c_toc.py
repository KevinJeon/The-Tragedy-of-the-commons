import numpy as np
import torch as tr
import torch.nn as nn
from torch.distributions import Categorical
class CPC(nn.Module):

    def __init__(self, num_action, num_channel):
        super(CPC, self).__init__()
        self.num_hidden = 128
        # Input Size (N, 88, 88, 3)
        self.encoder = nn.Sequential(
                nn.Conv2d(num_channel, 128 , kernel_size=11, stride=6, bias=False),
                nn.Conv2d(128, 64 , kernel_size=5, stride=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
         
        self.gru = nn.GRUCell(576, 128)
        for n, p in self.gru.named_parameters():
            if 'bias' in n:
                nn.init.constant_(p, 0)
            elif 'weight' in n:
                nn.init.orthogonal_(p)
        self.value = nn.Linear(128, 1)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(128, num_action) 
        
    def init_hidden(self, batch_size):
        return self.encoder.weight.new(1, 128).zero_()

    def forward(self, obs, h):
        '''
        x : (seq, c, h, w)
        h : (1, 1, hidden)
        '''
        bs = obs.size()[0]
        z = self.encoder(obs / 255.0).view(bs, -1)
        h = self.gru(z, h)
        s_f = self.linear(h)
        s_f = self.softmax(s_f)
        return self.value(h), s_f, h

class CPCAgent(object):

    def __init__(self, batch_size, seq_len, timestep, num_action, num_channel):
        super(CPCAgent, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.linear = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.action_encoder = nn.Embedding(num_action, 128)
        self.state_encoder = CPC(num_action, num_channel)
    def act(self, obs, h, is_train=True):
        obs = tr.from_numpy(obs).unsqueeze(0)
        h = h.unsqueeze(0)
        obs = obs.permute((0, 3, 1, 2))
        with tr.no_grad():
            v, s_f, h = self.state_encoder(obs, h)
            dist = Categorical(s_f)
            if is_train:
                act = dist.sample().unsqueeze(-1)
            else:
                act = dist.mode().unsqueeze(-1)
            a_f = self.action_encoder(act.view(-1))
            logprobs = dist.log_prob(act.unsqueeze(-1)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
            entropy = dist.entropy().mean()
            infos = [v, logprobs, h, s_f, a_f]
            return act, infos

    def evaluate(self, obs, h, act, mask):
        v, s_f, h = self.state_encoder(obs, h, mask)
        dist = Categorical(s_f)
        a_f = self.action_encoder(act.view(-1))
        logprobs = dist.log_probs(act)
        entropy = dist.entropy().mean()
        return v, logprobs, entropy, h, s_f, a_f

    def cpc(self, s_f, a_f):
        num_step, batch, num_hidden = a_f.shape
        s_a_f = s_f + a_f
        z_s = s_f[0].view(batch, num_hidden)
        z_a = s_f[0].view(batch, num_hidden)
        p_s = tr.empty(num_step, batch, num_hidden).float().to(self.device)
        p_a = tr.empty(num_step, batch, num_hidden).float().to(self.device)
        for i in range(self.num_step):
            s_linear = self.s_linear[i]
            p_s[i] = s_linear(z_s)
            a_linear = self.a_linear[i]
            p_a[i] = a_linear(z_a)

        nce_s = 0
        nce_a = 0
        acc_s = 0
        acc_a = 0
        for i in range(self.num_step):
            s_total = tr.mm(s_f[i], p_s[i].transpose(1, 0))
            a_total = tr.mm(a_f[i], p_a[i].transpose(1, 0))
            nce_s += tr.sum(tr.diag(self.log_softmax(s_total)))
            nce_a += tr.sum(tr.diag(self.log_softmax(a_total)))
            acc_s += tr.sum(tr.eq(tr.argmax(self.softmax(s_total), dim=0), tr.arange(0, batch).to(self.device)))
            acc_a += tr.sum(tr.eq(tr.argmax(self.softmax(a_total), dim=0), tr.arange(0, batch).to(self.device)))
        
        nce_s /= -1 * batch * self.num_step
        nce_a /= -1 * batch * self.num_step
        acc_s = 1. * acc_s.item() / (batch * num_step)
        acc_a = 1. * acc_a.item() / (batch * num_step)
        return acc_s, acc_a, nce_s, nce_a

    def train(self, samples, infos):
        num_obs = samples[0].size()[1:]
        num_act = samples[1].size()[-1]
        obss = samples[0][:-1].view(-1, *num_obs)
        hs = infos[2]
        acts = samples[1].view(-1, num_act)
        rews = samples[2]
        masks = samples[4][:-1]

        print('num_obs:{}, num_act:{}, obss: {}, acts: {}, rews: {}, masks: {}, hs: {}'.format(\
                num_obs, num_act, obss.size(), acts.size(), rews.size(), masks.size(), hs.size()))
        vs, logprobs, entropy, _, s_f, a_f = self.evaluate(obss, hs, acts, masks)
        vs = vs.view(num_step, num_sample, 1)
        s_f = s_f.view(num_step, num_sample, 1)
        a_f = a_f.view(num_step, num_sample, -1)
        logprobs = logprobs.view(num_step, num_sample, 1)
        adv = samples.ret[:-1] - vs
        vloss = (adv**2).mean()
        # nce_loss
        acc_s, acc_a, nce_s, nce_a = self.cpc(s_f, a_f)
        cpc_res = dict(nce_state=nce_s, nce_action=nce_a, acc_state=acc_s, acc_action=acc_a)
        aloss = -(adv.detach() * logprobs).mean()
        self.optimizer.zero_grad()
        (vloss * self.vloss_coef + aloss - entropy * self.entropy_coef + nce_s).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return vloss.item(), aloss.item(), entropy.item(), cpc_res

