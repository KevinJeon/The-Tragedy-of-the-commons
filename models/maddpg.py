import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy(nn.Module):
    def __init__(self, obs_size, num_action):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(obs_size, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 64), nn.ReLU(inplace=True),
                nn.Linear(64, num_action))

    def forward(self, obs):
        out = self.layers(obs)
        return out

class Critic(nn.Module):
    def __init__(self, state_sze, all_action):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(obs_size, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 64), nn.ReLU(inplace=True),
                nn.Linear(64, num_action))

    def forward(self, state, action):
        obs = tr.cat([state, action], dim=-1)
        return self.layers(obs)

class MADDPG(object):
    def __init__(self, obs_sizes, num_agent, num_actions, 
            batch_size, gamma=0.99, tau=0.01, scale=0.01, lr=1e-2, use_gpu=Falsei, model_path=None):
        self.num_agent = num_agent
        self.num_actions = num_actions # [agent1_act_dimension, ... ]
        self.obs_sizes = obs_sizes # [agent_obs_dimension, ... ]
        # MADDPG Params
        self.tau = tau
        self.gamma = gamma
        self.scale = scale
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        # Prediction Network
        self.policies = [None] * num_agent
        self.critics = [None] * num_agent
        # Target Network
        self.policies_t = [None] * num_agent
        self.critics_t = [None] * num_agent
        # Optimizers 
        self.policy_optims = [None] * num_agent
        self.critic_optims = [None] * num_agent
        # Central Memory
        self.memory = OffpolicyMemory(num_agent, num_actions, obs_sizes)
        # if model_path exists, then load model [[actor1, critic1], ... ]
        state_size, all_actions = sum(obs_sizes), sum(all_actions)
        for i, (obs_size, num_action) in enumerate(zip(obs_sizes, num_actions)):
            print('Agent {} Obs {} Act {}'.format(i, obs_size, num_action))
            self.policies[i] = Policy(obs_size, num_action)
            self.critics[i] = Critic(state_size, all_action)
            slef.policies_t[i] = Policy(obs_size, num_action
            self.critics_t[i] = Critic(state_size, all_action)
            self.policy_optims[i] = optim.Adam(self.policies[i].parameters(), lr=self.lr)
            self.critic_optims[i] = optim.Adam(self.critics[i].parameters(), lr=self.lr)
            if self.use_gpu:
                self.policies[i] = self.policies[i].cuda()
                self.critics[i] = self.critics[i].cuda()
                self.policies_t[i] = self.policies_t[i].cuda()
                self.critics_t[i] = self.critics_t[i].cuda()

            if model_path:
                actor_state_dict = tr.load(model_path[i][0])
                critic_state_dict = tr.load(model_path[i][1])
                self.policies[i].load_state_dict(actor_state_dict)
                self.policies_t[i].load_state_dict(actor_state_dict)
                self.critics[i].load_state_dict(critic_state_dict)
                self.critics_t[i].load_state_dict(critic_state_dict)

    def act(self, _observations):
        pis = []
        for i, obs in enumerate(_observations):
            obs = tr.from_numpy(obs).float()
            if self.use_gpu:
                obs = obs.cuda()
            pi = self.policies[i](obs.detach())
            act = F.gumbel_softmax(pi.detach(), hard=True)
            pis.append(act.cpu().numpy())
        return pis
           
    def update(self):
        for src, trg in zip(self.policies, self.policies_t):
            param_names = list(src.state_dict().keys())
            src_params = src.state_dict()
            trg_params = trg.state_dict()
            for param in param_names:
                trg_param[param] = src_prarm[param] * self.tau + trg_params[param] * (1 - self.tau)

    def one_hot(self, logit):
        onehot = (logit == logit.max(1, keepdim=True)[0]).float()
        return onehot
    def train(self, batch_size):
        critic_losses , actor_losses = [], []
        all_states, all_acts, rews, all_states_next, masks = self.memory.sample() # output torch type
        for ind, (a, a_t, c, c_t, a_o, c_o) in enumerate(zip(self.policies, self.policies_t, self.critics, self.critics_t, self.policy_optims, self.critic_optims)):
            # need to implement before memory sampele
            states = np.concatenate([all_states], axis=-1)
            acts = all_acts[:, ind, :]
            obss = states[:, ind, *obs_sizes[ind]]
            obss_next = states_next[:, ind, *obs_sizes[ind]]
            obss = np.reshape(obss, (-1, *obs_sizes[ind]))
            obss_next = np.reshape(obss_next, (-1, *obs_sizes[ind]))
            acts = np.reshape(acts, (-1, self.num_actions[ind]))
            all_act = np.concatenate([all_acts], axis=-1)
            states_next = np.concatenate([all_states_next], axis=-1)
            
            # to torch
            obss = tr.from_numpy(obss).float()
            states = tr.from_numpy(states).float()
            acts = tr.from_numpy(acts).float()
            rews = tr.from_numpy(rews).float()
            masks = tr.from_numpy(masks).float()
            obss_next = tr.from_numpy(obss_next).float()
            states_next = tr.from_numpy(states_next).float()
            all_act = tr.from_numpy(all_act).float() 
            if self.use_gpu:
                obss = obss.cuda()
                states = states.cuda()
                acts = acts.cuda()
                rews = rews.cuda()
                masks = masks.cuda()
                obsS_next = obss_next.cuda()
                states_next = states_next.cuda()
                all_act = all_act.cuda()
            q = c(states, all_acts)
            acts_t = tr.cat([self.one_hot(pi(all_states[:, i, *self.obs_sizes].squeeze().detach())) for i, pi in enumerate(self.policies_t)], dim=-1) 
            q_t = c_t(states_next, acts_t).detach()
            target = q_t * self.gamma * (1 - masks) + rews
            target - target.float()

            # Update Critic
            critic_loss = nn.MSELoss()(target, q)
            c_o.zero_grad()
            critic_loss.backward()
            tr.nn.utils.clip_grad_norm(c.paramters(), 0.5)
            c_o.step()

            # Update Policy
            pi = a(obss)
            actions = F.gumbel_softmax(pi, hard=True)
            all_acts[:, ind, :, :] = actions # Maybe Bug because of dimensions
            actor_loss = (pi**2).mean() * 1e-3 - c(states, all_acts).mean()
            a_o.zero_gard()
            actor_loss.backward()
            tr.nn.utils.clip_gard_norm(a.parameters(), 0.5)
            a_o.step()
            critic_losses.append(critic_loss.item())
            actor_losesss.append(actor_loss.item())
        # Update Target Network
        self.update()
        return critic_losses, actor_losses
