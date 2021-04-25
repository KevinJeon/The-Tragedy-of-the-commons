import numpy as np
import torch as tr
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

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
         
        self.gru = nn.GRU(576, 128)
        for n, p in self.gru.named_parameters():
            if 'bias' in n:
                nn.init.constant_(p, 0)
            elif 'weight' in n:
                nn.init.orthogonal_(p)
        self.value = nn.Linear(128, 1)
        self.softmax = nn.Softmax()
        
    def init_hidden(self, batch_size):
        return self.encoder.weight.new(1, 128).zero_()

    def forward(self, obs, h, bs=1):
        '''
        x : (seq, c, h, w)
        h : (1, 1, hidden)
        '''
        step, c, he, w = obs.size()
        z = self.encoder(obs.view(-1, c, he, w) / 255.0).view(bs, step, -1)
        s_f, h = self.gru(z, h)
        s_f = s_f.view(bs*step, -1)
        h = h.squeeze(0)
        return self.value(h), s_f, h

class CPCAgent(object):

    def __init__(self, batch_size, seq_len, timestep, num_action, num_channel):
        super(CPCAgent, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.linear = nn.ModuleList([nn.Linear(128, 512) for i in range(timestep)])
        self.action_encoder = nn.Embedding(num_action, 128)
        self.state_encoder = CPC(num_action, num_channel)
        self.linear = nn.Linear(128, num_action)
    def act(self, obs, h, is_train=True):
        obs = tr.from_numpy(obs).unsqueeze(0)
        h = h.unsqueeze(0)
        obs = obs.permute((0, 3, 1, 2))
        with tr.no_grad():
            v, s_f, h = self.state_encoder(obs, h.unsqueeze(0))
            lin_s_f = self.linear(s_f)
            lin_s_f = F.softmax(lin_s_f)
            dist = Categorical(lin_s_f)
            if is_train:
                act = dist.sample().unsqueeze(-1)
            else:
                act = dist.mode().unsqueeze(-1)
            a_f = self.action_encoder(act.view(-1))
            logprobs = dist.log_prob(act.squeeze(-1)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
            entropy = dist.entropy().mean()
            infos = [v, logprobs, h, s_f, a_f]
            return act, infos

    def evaluate(self, obs, h, act, mask):
        print(obs.size(), h.size(), act.size(), mask.size())
        v, s_f, h = self.state_encoder(obs, h)
        lin_s_f = self.linear(s_f)
        lin_s_f = F.softmax(lin_s_f)
        dist = Categorical(lin_s_f)
        a_f = self.action_encoder(act.view(-1).long())
        print(act.size(), lin_s_f.size())
        logprobs = dist.log_prob(act.squeeze(-1)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
        print(logprobs.size())
        entropy = dist.entropy().mean()
        return v, logprobs, entropy, h, s_f, a_f

    def cpc(self, s_f, a_f):
        print(s_f.size(), a_f.size())
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
        # Please Check for indices of samples
        # (batch, step, *shapes)
        num_act = samples[1].size()[-1]
        bs, step = samples[2].size()[:2]
        obss = samples[0][:, :-1].permute((0, 1, 4, 2, 3))
        num_obs = obss.size()[2:]
        hs = infos[2][:, :-1]
        acts = samples[1]
        rews = samples[2]
        rets = samples[3]
        masks = samples[4][:, :-1]

        print('num_obs:{}\n, num_act:{}\n, obss: {}\n, acts: {}\n, rews: {}\n, masks: {}\n, hs: {}\n'.format(\
                num_obs, num_act, obss.size(), acts.size(), rews.size(), masks.size(), hs.size()))
        vs, logprobs, entropy, _, s_f, a_f = self.evaluate(obss.reshape(-1, *num_obs), hs[:, 0, :].view(-1, 128), acts.view(-1, num_act), masks.reshape(-1, 1))
        vs = vs.view(step, bs, 1)
        s_f = s_f.view(step, bs, -1)
        a_f = a_f.view(step, bs, -1)
        logprobs = logprobs.view(step, bs, 1)
        adv = rets[:, :-1] - vs
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

