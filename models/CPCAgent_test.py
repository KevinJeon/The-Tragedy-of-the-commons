from models.Agent import Agent

import os

import torch
import torch as tr
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.sys import make_dir


class CPC(nn.Module):

    def __init__(self, num_action, num_channel, batch_size, name='ra'):
        super(CPC, self).__init__()
        self.num_hidden = 128
        # Input Size (N, 88, 88, 3)
        if name == 'ra':
            lin_in = 46464
        else:
            lin_in = 10816
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 6, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(lin_in, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(128, 128)
        for n, p in self.gru.named_parameters():
            if 'bias' in n:
                nn.init.constant_(p, 0)
            elif 'weight' in n:
                nn.init.orthogonal_(p)
        self.value = nn.Linear(128, 1)
        self.policy = nn.Linear(128, num_action)
        self.softmax = nn.Softmax()
        self.batch_size = batch_size
        self.name = name
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
        z = self.encoder(obs.reshape(-1, c, he, w)).reshape(bs, step, -1)
        c, h = self.gru(z, h)
        c = c.view(bs * step, -1)
        h = h.squeeze(0)
        return self.value(c), z.view(-1, 128), c, self.policy(c), h


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
        self.state_encoder = CPC(action_dim, num_channel, batch_size, name).to(self.device)

        self.optimizer = optim.RMSprop(self.parameters(), lr, eps=eps, alpha=alpha)


        self.max_grad_norm = max_grad_norm
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.softmax = nn.Softmax(dim=0)
        self.entropy_coef = entropy_coef
        self.v_loss_coef = v_loss_coef

    def act(self, obs, h, sample=True):
        obs = tr.from_numpy(obs).unsqueeze(0)
        h = h.unsqueeze(0).to(self.device)
        obs = obs.permute((0, 3, 1, 2)).to(self.device)
        with tr.no_grad():
            v, z, c , pi, h = self.state_encoder(obs, h.unsqueeze(0))
            lin_s_f = F.softmax(pi, dim=-1)
            dist = Categorical(lin_s_f)

            if sample:
                act = dist.sample().unsqueeze(-1)
            else:
                act = dist.probs.argmax(dim=-1, keepdim=True)

        logprobs = dist.log_prob(act.squeeze(-1)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().mean()
        infos = [v, logprobs, h]
        return act, infos

    def evaluate(self, obs, h, act, mask):
        h = h.unsqueeze(0).to(self.device)
        v, z, c , pi ,h = self.state_encoder(obs.to(self.device), h.to(self.device))
        lin_s_f = F.softmax(pi, dim=-1)
        dist = Categorical(lin_s_f)
        logprobs = dist.log_prob(act.squeeze(-1).to(self.device)).view(act.size(0), -1).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().mean()
        return v, logprobs, entropy, h, z, c

    def get_value(self, obs, h):
        obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2).to(self.device)
        v, _, _ , _ ,_ = self.state_encoder(obs, h.view(1, 1, -1).to(self.device))
        return v

    def cpc_loss(self, z, c):
        num_step, batch_size, num_hidden = z.shape
        nce = 0
        acc = 0
        for i in range(num_step):
            mm_total = tr.mm(z[i], c[i].transpose(1, 0))
            nce += tr.sum(tr.diag(self.log_softmax(mm_total)))
            acc += tr.sum(tr.eq(tr.argmax(self.softmax(mm_total), dim=0), tr.arange(0, batch_size).to(self.device)))
        nce /= -1 * batch_size * num_step
        acc = 1. * acc.item() / (batch_size * num_step)
        return acc, nce

    def train(self, samples, infos):
        # Please Check for indices of samples
        # (batch, step, *shapes)
        num_act = samples[1].size()[-1]
        step, bs = samples[2].size()[:2]
        obss = samples[0].permute((0, 1, 4, 2, 3))
        num_obs = obss.size()[2:]
        hs = infos[2]
        acts = samples[1]
        act_inds = samples[2]
        rews = samples[3]
        rets = samples[4].to(self.device)
        masks = samples[5]
        vs, logprobs, entropy, _, z, c = self.evaluate(obss.reshape(-1, * num_obs), hs[:, 0, :].reshape(-1, 128),
                                                                 act_inds.reshape(-1, 1), masks.reshape(-1, 1))
        vs = vs.view(step, bs, 1)
        z = z.view(step, bs, -1)
        c = c.view(step, bs, -1)
        logprobs = logprobs.view(step, bs, 1)

        adv = rets - vs
        vloss = (adv ** 2).mean()
        # nce_loss
        acc, nce = self.cpc_loss(z, c)
        cpc_res = None
        aloss = -(adv.detach() * logprobs).mean()
        self.optimizer.zero_grad()
        (vloss * self.v_loss_coef + aloss - entropy * self.entropy_coef + nce).backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return vloss.item(), aloss.item(), entropy.item(), nce.item()

    def save(self, num):
        make_dir('models')
        make_dir(f'models/{num}')

        torch.save(self.state_dict(), os.path.join('.', 'models', str(num), f'{self.name}.pth'))


class CPCAgentGroup(object):

    def __init__(self,
                 name,
                 lr,
                 num_agent,
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
                 agent_name,
                 device):
        super(CPCAgentGroup, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.device = device
        self.agent_name = agent_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_cpc = use_cpc
        self.agents = [A2CCPCAgent(agent_name,
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
                                   device) for i in range(num_agent)]

        self.seq_len = seq_len
        self.logger = None

    def act(self, memory, obses, step, sample=True):
        actions = []
        infos = []
        for obs, agent, h in zip(obses, self.agents, memory.h):
            act, info = agent.act(obs, h[memory.step - 1, memory.n], sample=sample)
            actions.append(act.view(-1).detach().cpu().numpy())
            infos.append(info)

        return actions, infos

    def train(self, memory, logger=None, total_step=None):
        with tr.no_grad():
            next_vals = []
            for i, agent in enumerate(self.agents):
                next_val = agent.get_value(memory.obs[i][-1, memory.n - 1], memory.h[i][-1, memory.n - 1]).detach()

                next_vals.append(next_val)
        memory.compute_return(next_vals, gamma=0.99)
        memory.n += 1
        if memory.n % self.batch_size == 0:  # if (memory.n != 0) and (memory.n % args.batch_size == 0):
            v_losses, a_losses, entropies, nces = [], [], [], []
            for i, agent in enumerate(self.agents):
                # check for return calculate
                obss, acts, act_inds, rews, rets, masks = \
                    memory.obs[i], memory.act[i], memory.act_ind[i], memory.rew[
                    i], memory.ret[i], memory.mask[i]
                if self.use_cpc:
                    vs, logprobs, hs = memory.val[i], memory.logprob[i], memory.h[i]
                avg_v_loss = 0
                avg_a_loss = 0
                avg_entropy = 0
                avg_nce = 0
                for j in range(10):
                    samples = (obss[j:(j+1)*100], acts[j:(j+1)*100], act_inds[j:(j+1)*100],
                               rews[j:(j+1)*100], rets[j:(j+1)*100], masks[j:(j+1)*100])
                    infos = (vs[j:(j+1)*100], logprobs[j:(j+1)*100], hs[j:(j+1)*100])

                    v_loss, a_loss, entropy, nce = agent.train(samples, infos)
                    avg_v_loss += v_loss
                    avg_a_loss += a_loss
                    avg_entropy += entropy
                    avg_nce += nce
                if logger:
                    if self.agent_name == 'ma':
                        logger.log('agent_{0}/train/v_loss'.format(4), avg_v_loss / (j + 1), total_step)
                        logger.log('agent_{0}/train/a_loss'.format(4), avg_a_loss / (j + 1), total_step)
                        logger.log('agent_{0}/train/entropy'.format(4), avg_entropy / (j + 1), total_step)
                        logger.log('agent_{0}/train/nce'.format(4), avg_nce / (j + 1), total_step)
                        # writer.add_scalar('agent_{0}/train/cpc_res'.format(i), cpc_res / agent_count, total_step)
                    else:
                        logger.log('agent_{0}/train/v_loss'.format(i), avg_v_loss / (j + 1), total_step)
                        logger.log('agent_{0}/train/a_loss'.format(i), avg_a_loss / (j + 1), total_step)
                        logger.log('agent_{0}/train/entropy'.format(i), avg_entropy / (j + 1), total_step)
                        logger.log('agent_{0}/train/nce'.format(i), avg_nce / (j + 1), total_step)
                v_losses.append(avg_v_loss)
                a_losses.append(avg_a_loss)
                entropies.append(avg_entropy)
                nces.append(avg_nce)
            if self.agent_name == 'ma':
                logger.log('agent_{0}/train/v_loss'.format(4), sum(v_losses) / len(self.agents), total_step)
                logger.log('agent_{0}/train/a_loss'.format(4), sum(a_losses) / len(self.agents), total_step)
                logger.log('agent_{0}/train/entropy'.format(4), sum(entropies) / len(self.agents), total_step)
                logger.log('agent_{0}/train/entropy'.format(4), sum(nces) / len(self.agents), total_step)
            else:
                logger.log('train/v_loss'.format(0), sum(v_losses) / len(self.agents), total_step)
                logger.log('train/a_loss'.format(0), sum(a_losses) / len(self.agents), total_step)
                logger.log('train/entropy'.format(0), sum(entropies) / len(self.agents), total_step)
                logger.log('train/entropy'.format(0), sum(nces) / len(self.agents), total_step)
            #v_losses, a_losses, entropies, nces = None, None, None, None
            memory.after_update()  # Need to Check!
    def save(self, model_num):
        for agent in self.agents:
            agent.save(model_num)
