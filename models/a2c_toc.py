import numpy as np
import torch as tr
import torch.nn as nn

class CPC(nn.Module):

    def __init__(self, timestep, batch_size, seq_len, num_channel):
        super(CPC, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timetep = timestep
        self.num_hidden = 128
        # Input Size (N, 88, 88, 3)
        self.encoder = nn.Sequential(
                nn.Conv2d(num_chaanel, 512 , kernel_size=10, stride=5, padding=3, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True))
        self.lstm = nn.LSTM(512, 128, num_layers=1, batch_first=True)
        self.linear = nn.Modulelist([nn.Linear(128, 512) for i in range(timestep)])
        self.policy = nn.Linear(128, num_action)
        self.value = nn.Linear(128, 1)
    def init_hidden(self, batch_size):
        return tr.zeros(1, batch_size, 128)

    def forward(self, obss, h, acts):
        '''
        x : (seq, batch, c, h, w)
        h : (1, batch, hidden)
        act : (seq, batch, num_action)
        '''
        seq, batch, c, h, w = obss.size()
        # random t
        obs = obss.view(seq * batch, c, h, w)
        t = tr.randint(self.seq_len - self.timestep, size=(1,)).long() # range for 0 ~ seq - timestep
        z = self.encoder(obss)
        z_ts = tr.empty((self.timestep, batch, 512))
        for i in range(self.timestep):
            z_ts[i] = z[: , t+i, :].view(batch, 512)
        z_lstm = []
        for i in range(self.seq_len + 1):
            out, h = self.lstm(z[i], h)
            z_lstm.append(out)
        c_t = z_lstm[t].squeeze(1)
        c = tr.stack(tensors=z_lstm, dim=0)
        pi = self.policy(c)
        v = self.value(c)
        pred = tr.empty((self.timestep, batch, 512)).float()
        for i in range(self.timestep):
            linear = self.linear[i]
            pred[i] = linear(c_t)
        nce = 0
        for i in range(self.timestep):
            loss = tr.mm(z_ts[i], pred[i].transpose(1,0))
            nce += tr.sum(tr.diag(F.log_softmax(loss)))
        nce /= -1 * batch * self.timestep
        return nce, hidden, pi, v
    
    def pi(self, obs, h):
        '''
        x : (seq, c, h, w)
        h : (1, 1, hidden)
        '''
        z = self.encoder(obs)
        c, hidden = self.lstm(z)
        dist = self.policy(c)
        return act, h

    def v(self, obs, h):
        z = self.encoder(obs)
        c, hidden = self.lstm(z, h)
        v = self.value(c)
        return v
class CPCAgent(object):

    def __init__(self):
        super(CPCAgent, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model = CPC()

    def act(self, obs, h, is_train=True):
        v, s_f, h = self.state_encoder(obs, h)
        dist = Categorical(s_f)
        if is_train:
            act = dist.sample()
        else:
            act = dist.mode()
        a_f = self.action_encoder(act.view(-1))
        logprobs = dist.log_probs(pi)
        entropy dist.entropy().mean()
        return v, act, logprobs, h, s_f, a_f

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

    def train(self, samples):
        num_obs = samples.obs.size()[2:]
        num_act = samples.act.size()[-1]
        num_step, num_sample, _ = samples.rew.size()
        obss = samples.obs[:-1].view(-1, *num_obs)
        hs = samples.h[0].view(-1, self.model.h.size())
        acts = samples.act.view(-1, num_act)
        rews = samples.rews
        masks = samples.masks[:-1]
        vs, logprobs, entropy, _, s_f, a_f = self.evaluate(obss, hs, acts, masks)
        vs = vs.view(num_step, num_sample, 1)
        s_f = s_f.view(num_step, num_sample, 1)
        a_f = a_f.view(num_step, num_sample, -1)
        logprobs = logprobs.view(num_step, num_sample, 1)
        adv = samples.ret[:-1] - vs
        vloss =  adv**2.mean()
        # nce_loss
        acc_s, acc_a, nce_s, nce_a = self.cpc(s_f, a_f)
        cpc_res = dict(nce_state=nce_s, nce_action=nce_a, acc_state=acc_s, acc_action=acc_a)
        aloss = -(adv.detach() * logprobs).mean()
        self.optimizer.zero_grad()
        (vloss * self.vloss_coef + aloss - entropy * self.entropy_coef + nce_s).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return vloss.item(), aloss.item(), entropy.item(), cpc_res

