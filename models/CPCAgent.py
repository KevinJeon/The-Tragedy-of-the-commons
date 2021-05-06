from models.Agent import Agent

import torch as tr
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim


class CPC(nn.Module):

    def __init__(self, num_action, num_channel, batch_size):
        super(CPC, self).__init__()
        self.num_hidden = 128
        # Input Size (N, 88, 88, 3)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 128, kernel_size=11, stride=6, bias=False),
            nn.Conv2d(128, 64, kernel_size=5, stride=3, bias=False),
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
        self.batch_size = batch_size

    def init_hidden(self, batch_size):
        return self.encoder.weight.new(1, 128).zero_()

    def forward(self, obs, h, bs=1):
        '''
        x : (seq, c, h, w)
        h : (1, 1, hidden)
        '''
        n, c, he, w = obs.size()
        if n == 1:
            bs = 1
            step = 1
        else:
            bs = self.batch_size
            step = int(n // self.batch_size)

        z = self.encoder(obs.reshape(-1, c, he, w) / 255.0).reshape(bs, step, -1)
        s_f, h = self.gru(z, h)
        s_f = s_f.view(bs * step, -1)
        h = h.squeeze(0)
        return self.value(s_f), s_f, h


class A2CCPCAgent(nn.Module):

    def __init__(self,
                 name,
                 lr,
                 eps,
                 alpha,
                 max_grad_norm,
                 entropy_coef,
                 v_loss_coef,
                 obs_dim,
                 action_dim,
                 batch_size,
                 seq_len,
                 num_channel,
                 use_cpc,
                 device
                 ):
        super(A2CCPCAgent, self).__init__()

        self.name = name
        self.batch_size = batch_size
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_cpc = use_cpc
        self.eps = eps
        self.alpha = alpha

        ''' Networks '''
        self.s_linear = nn.ModuleList([nn.Linear(128, 128) for _ in range(seq_len)]).to(self.device)
        self.a_linear = nn.ModuleList([nn.Linear(128, 128) for _ in range(seq_len)]).to(self.device)
        self.action_encoder = nn.Embedding(action_dim, 128).to(self.device)
        self.state_encoder = CPC(action_dim, num_channel, batch_size).to(self.device)
        self.act_linear = nn.Linear(128, action_dim).to(self.device)

        self.optimizer = optim.RMSprop(self.parameters(), lr, eps=eps, alpha=alpha)


        self.max_grad_norm = max_grad_norm
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        self.entropy_coef = entropy_coef
        self.v_loss_coef = v_loss_coef

    def act(self, obs, h, sample=False):
        obs = tr.from_numpy(obs).unsqueeze(0)
        h = h.unsqueeze(0).to(self.device)
        obs = obs.permute((0, 3, 1, 2)).to(self.device)

        with tr.no_grad():
            v, s_f, h = self.state_encoder(obs, h.unsqueeze(0))
            lin_s_f = self.act_linear(s_f)
            lin_s_f = F.softmax(lin_s_f, dim=-1)
            dist = Categorical(lin_s_f)
            if sample:
                act = dist.sample().unsqueeze(-1)
            else:
                # TODO change sample() to mode()
                # act = dist.mode().unsqueeze(-1)
                act = dist.sample().unsqueeze(-1)
            a_f = self.action_encoder(act.view(-1))

        logprobs = dist.log_prob(act.squeeze(-1)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().mean()
        infos = [v, logprobs, h, s_f, a_f]

        return act, infos

    def evaluate(self, obs, h, act, mask):
        h = h.unsqueeze(0).to(self.device)
        v, s_f, h = self.state_encoder(obs.to(self.device), h.to(self.device))
        lin_s_f = self.act_linear(s_f)
        lin_s_f = F.softmax(lin_s_f, dim=-1)
        dist = Categorical(lin_s_f)
        a_f = self.action_encoder(act.view(-1).long().to(self.device))
        logprobs = dist.log_prob(act.squeeze(-1).to(self.device)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().mean()
        return v, logprobs, entropy, h, s_f, a_f

    def get_value(self, obs, h):
        obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2).to(self.device)
        v, _, _ = self.state_encoder(obs, h.view(1, 1, -1).to(self.device))
        return v

    def cpc(self, s_f, a_f):
        num_step, batch_size, num_hidden = a_f.shape
        s_a_f = s_f + a_f
        z_s = s_f[0].view(batch_size, num_hidden)
        z_a = s_a_f[0].view(batch_size, num_hidden)
        p_s = tr.empty(num_step, batch_size, num_hidden).float().to(self.device)
        p_a = tr.empty(num_step, batch_size, num_hidden).float().to(self.device)
        for i in range(num_step):
            s_linear = self.s_linear[i]
            p_s[i] = s_linear(z_s)
            a_linear = self.a_linear[i]
            p_a[i] = a_linear(z_a)

        return p_s, p_a

    def cpc_loss(self, s_f, a_f):
        num_step, batch_size, num_hidden = a_f.shape
        nce_s = 0
        nce_a = 0
        acc_s = 0
        acc_a = 0
        s_a_f = s_f + a_f
        p_s, p_a = self.cpc(s_f, a_f)
        for i in range(num_step):
            s_total = tr.mm(s_f[i], p_s[i].transpose(1, 0))
            a_total = tr.mm(s_a_f[i], p_a[i].transpose(1, 0))
            nce_s += tr.sum(tr.diag(self.log_softmax(s_total)))
            nce_a += tr.sum(tr.diag(self.log_softmax(a_total)))
            acc_s += tr.sum(tr.eq(tr.argmax(self.softmax(s_total), dim=0), tr.arange(0, batch_size).to(self.device)))
            acc_a += tr.sum(tr.eq(tr.argmax(self.softmax(a_total), dim=0), tr.arange(0, batch_size).to(self.device)))

        nce_s /= -1 * batch_size * num_step
        nce_a /= -1 * batch_size * num_step
        acc_s = 1. * acc_s.item() / (batch_size * num_step)
        acc_a = 1. * acc_a.item() / (batch_size * num_step)
        return acc_s, acc_a, nce_s, nce_a

    def train(self, samples, infos):
        # Please Check for indices of samples
        # (batch, step, *shapes)
        num_act = samples[1].size()[-1]
        step, bs = samples[2].size()[:2]
        obss = samples[0][:-1].permute((0, 1, 4, 2, 3))
        num_obs = obss.size()[2:]
        hs = infos[2][:-1]
        acts = samples[1]
        act_inds = samples[2]
        rews = samples[3]
        rets = samples[4].to(self.device)
        masks = samples[5][:-1]

        vs, logprobs, entropy, _, s_f, a_f = self.evaluate(obss.reshape(-1, *num_obs), hs[:, 0, :].reshape(-1, 128),
                                                                 act_inds.reshape(-1, 1), masks.reshape(-1, 1))
        vs = vs.view(step, bs, 1)
        s_f = s_f.view(step, bs, -1)
        a_f = a_f.view(step, bs, -1)
        logprobs = logprobs.view(step, bs, 1)

        adv = rets[:-1] - vs
        vloss = (adv ** 2).mean()
        # nce_loss
        acc_s, acc_a, nce_s, nce_a = self.cpc_loss(s_f, a_f)
        cpc_res = dict(nce_state=nce_s, nce_action=nce_a, acc_state=acc_s, acc_action=acc_a)
        aloss = -(adv.detach() * logprobs).mean()
        self.optimizer.zero_grad()
        (vloss * self.v_loss_coef + aloss - entropy * self.entropy_coef + nce_s).backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return vloss.item(), aloss.item(), entropy.item(), cpc_res


class CPCAgentGroup(object):

    def __init__(self,
                 name,
                 agent_types,
                 lr,
                 eps,
                 alpha,
                 max_grad_norm,
                 entropy_coef,
                 v_loss_coef,
                 obs_dim,
                 action_dim,
                 batch_size,
                 seq_len,
                 num_channel,
                 use_cpc,
                 device):
        super(CPCAgentGroup, self).__init__()

        self.name = name
        self.batch_size = batch_size
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_cpc = use_cpc

        self.agent_types = agent_types
        self.agents = [A2CCPCAgent(name,
                                   lr,
                                   eps,
                                   alpha,
                                   max_grad_norm,
                                   entropy_coef,
                                   v_loss_coef,
                                   obs_dim,
                                   action_dim,
                                   batch_size,
                                   seq_len,
                                   num_channel,
                                   use_cpc,
                                   device) for _ in self.agent_types]

        self.seq_len = seq_len

    def act(self, memory, obses, step, sample=False):
        actions = []
        infos = []

        for obs, agent, h in zip(obses, self.agents, memory.h):
            act, info = agent.act(obs, h[step, memory.n], sample=sample)
            actions.append(act.view(-1).detach().cpu().numpy())
            infos.append(info)

        return actions, infos

    def train(self, memory, logger=None, total_step=None):

        with tr.no_grad():
            next_vals = []
            for i, agent in enumerate(self.agents):
                next_val = agent.get_value(memory.obs[i][-1, memory.n], memory.h[i][-1, memory.n]).detach()
                next_vals.append(next_val)

        memory.compute_return(next_vals, gamma=0.99)
        memory.n += 1
        if memory.n % self.batch_size == 0:  # if (memory.n != 0) and (memory.n % args.batch_size == 0):
            v_losses, a_losses, entropies = [], [], []
            for i, agent in enumerate(self.agents):
                # check for return calculate
                obss, acts, act_inds, rews, rets, masks = \
                    memory.obs[i], memory.act[i], memory.act_ind[i], memory.rew[
                    i], memory.ret[i], memory.mask[i]
                samples = (obss, acts, act_inds, rews, rets, masks)
                infos = None

                if self.use_cpc:
                    vs, logprobs, hs, s_fs, a_fs = \
                        memory.val[i], memory.logprob[i], memory.h[i], memory.s_feat[i], \
                                                   memory.a_feat[i]
                    infos = (vs, logprobs, hs, s_fs, a_fs)

                v_loss, a_loss, entropy, cpc_res = agent.train(samples, infos)

                agent_count = len(self.agents)
                if logger:
                    logger.log('agent_{0}/train/v_loss'.format(i), v_loss, total_step)
                    logger.log('agent_{0}/train/a_loss'.format(i), a_loss, total_step)
                    logger.log('agent_{0}/train/entropy'.format(i), entropy, total_step)
                    # writer.add_scalar('agent_{0}/train/cpc_res'.format(i), cpc_res / agent_count, total_step)

                v_losses.append(v_loss)
                a_losses.append(a_loss)
                entropies.append(entropy)
                memory.after_update()  # Need to Check!

            # anum = args.red + args.blue

            logger.log('train/v_loss', sum(v_losses) / len(self.agents), total_step)
            logger.log('train/a_loss', sum(a_losses) / len(self.agents), total_step)
            logger.log('train/entropy', sum(entropies) / len(self.agents), total_step)

            # print('Value loss : {:.2f} Action loss : {:.2f} Entropy : {:.2f}'.format(sum(vlosses) / anum,
                                                                                     # sum(alosses) / anum,
                                                                                     # sum(entropies) / anum))
