import cv2 as cv
import argparse
from tocenv.env import TOCEnv
from models.a2c_toc import CPCAgent, CPCTrainer
from models.rulebased import RuleBasedAgent
from tocenv.utils.storage import RolloutStorage
import os
import torch as tr
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import datetime

AGENT_TYPE = dict(rule=[RuleBasedAgent, None], ac=[CPCAgent, CPCTrainer])
AGENT_CONFIG = dict(rule=[dict(prefer=None, obs_size=None), None],
        ac=[dict(batch_size=None, seq_len=None, num_action=8, num_channel=3), \
                dict(eps=1e-5, alpha=0.99, lr=5e-4, max_grad_norm=0.5, vloss_coef=0.5, entropy_coef=0.01)])

def parse_args():
    parser = argparse.ArgumentParser(description='ToC params')
    parser.add_argument('--blue', default=5, type=int)
    parser.add_argument('--red', default=5, type=int)
    parser.add_argument('--num_episode', default=10000, type=int)
    parser.add_argument('--max_step', '-m', default=200, type=int)
    parser.add_argument('--agent_type', '-a', default='ac', type=str)
    parser.add_argument('--batch_size', '-bs', default=32, type=int)
    parser.add_argument('--save_freq', '-sf', default=32, type=int)
    parser.add_argument('--base_dir', '-b', default='./logs', type=str) 
    parser.add_argument('--tensorboard', action="store_true")
    parser.add_argument('--use_gpu', default=False, type=bool)
    parser.add_argument('--load_dir', default=None, type=str)
    args = parser.parse_args()
    return args

def select_actions(obss, agents, step, hs=None, n=None):
    actions = []
    infos = []
    if hs == None:
        hs = [None] * len(agents)
    for obs, agent, h in zip(obss, agents, hs):
        act, info = agent.act(obs, h[step, n])
        actions.append(act.view(-1).numpy())
        infos.append(info)
    return actions, infos

def main(args):
    base_dir = osp.join(args.base_dir,datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) 
    save_dir = osp.join(base_dir, 'save')
    log_dir = osp.join(base_dir, 'log')
    if not osp.exists(base_dir):
        os.mkdir(base_dir)
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        if not osp.exists(log_dir):
            os.mkdir(log_dir)
    if args.tensorboard:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    prefer = ['blue']*args.blue+['red']*args.red
    env = TOCEnv(agents=prefer, apple_color_ratio=0.5, apple_spawn_ratio=0.1)
    if args.agent_type == 'ac':
        AGENT_CONFIG[args.agent_type][0]['batch_size'] = args.batch_size
        AGENT_CONFIG[args.agent_type][0]['seq_len'] = args.max_step
    agents = [AGENT_TYPE[args.agent_type][0](**AGENT_CONFIG[args.agent_type][0]) for i in range(args.blue + args.red)]
    if args.load_dir:
        for i, agent in enumerate(agents):
            agent.load_state_dict(tr.load(args.load_dir))
    trainers = [AGENT_TYPE[args.agent_type][1](agent=agents[i], **AGENT_CONFIG[args.agent_type][1]) for i in range(args.blue + args.red)]
    env.obs_type = 'rgb_array'
    memory = RolloutStorage(agent_type='ac',num_agent=args.blue+args.red, num_step=args.max_step, \
            batch_size=args.batch_size, num_obs=(88, 88, 3), num_action=8, num_rec=128)

    total_step = 0

    for ep in range(args.num_episode):
        obss, color_agents = env.reset()
        #key = cv.waitKey(0) to debug
        epi_rews = []
        epi_rew = 0
        for step in range(args.max_step):
            image = env.render()
            cv.imshow('Env', image)
            key = cv.waitKey(1)
            sampled_action = []
            sampled_actions, infos = select_actions(obss, agents, step, memory.h, memory.n)
            obss_next, rews, masks, env_info = env.step(actions=sampled_actions)
            memory.add(obss, sampled_actions, rews, masks, infos)
            obss = obss_next
            total_step += 1
            epi_rew += sum(rews)
        ''' Log Environment Statistics '''
        epi_rews.append(epi_rew)
        if writer:
            writer.add_scalar('train/episode', ep, total_step)
            statistics = env_info['statistics']

            writer.add_scalar('statistics/movement/move', statistics['movement']['move'], total_step)
            writer.add_scalar('statistics/movement/rotate', statistics['movement']['rotate'], total_step)
            writer.add_scalar('statistics/punishment/punishing', statistics['punishment']['punishing'], total_step)
            writer.add_scalar('statistics/punishment/punished', statistics['punishment']['punished'], total_step)
            writer.add_scalar('statistics/punishment/valid_rate', statistics['punishment']['valid_rate'], total_step)
            writer.add_scalar('statistics/eaten_apples/total/blue', statistics['eaten_apples']['total']['blue'], total_step)
            writer.add_scalar('statistics/eaten_apples/total/red', statistics['eaten_apples']['total']['red'], total_step)
            writer.add_scalar('statistics/eaten_apples/team_blue/blue', statistics['eaten_apples']['team']['blue']['blue'], total_step)
            writer.add_scalar('statistics/eaten_apples/team_blue/red', statistics['eaten_apples']['team']['blue']['red'], total_step)
            writer.add_scalar('statistics/eaten_apples/team_red/red', statistics['eaten_apples']['team']['red']['red'], total_step)
            writer.add_scalar('statistics/eaten_apples/team_red/blue', statistics['eaten_apples']['team']['red']['blue'], total_step)


        with tr.no_grad():
            next_vals = []
            for i, agent in enumerate(agents):
                next_val = agent.get_value(memory.obs[i][-1, memory.n], memory.h[i][-1, memory.n]).detach()
                next_vals.append(next_val)

        memory.compute_return(next_vals, gamma=0.99)
        memory.n += 1
        if memory.n % args.batch_size == 0: # if (memory.n != 0) and (memory.n % args.batch_size == 0):
            vlosses, alosses, entropies = [], [], []
            for i, trainer in enumerate(trainers):
                # check for return calculate
                obss, acts, act_inds, rews, rets, masks = memory.obs[i], memory.act[i], memory.act_ind[i], memory.rew[i], memory.ret[i], memory.mask[i]
                samples = (obss, acts, act_inds, rews, rets, masks)
                infos = None
                if args.agent_type == 'ac':
                    vs, logprobs, hs, s_fs, a_fs = memory.val[i], memory.logprob[i], memory.h[i], memory.s_feat[i], memory.a_feat[i]
                    infos = (vs, logprobs, hs, s_fs, a_fs)

                vloss, aloss, entropy, cpc_res = trainer.train(samples, infos)

                agent_count = args.red + args.blue
                if writer:
                    writer.add_scalar('agent_{0}/train/v_loss'.format(i), vloss / agent_count, total_step)
                    writer.add_scalar('agent_{0}/train/a_loss'.format(i), aloss / agent_count, total_step)
                    writer.add_scalar('agent_{0}/train/entropy'.format(i), entropy / agent_count, total_step)
                    # writer.add_scalar('agent_{0}/train/cpc_res'.format(i), cpc_res / agent_count, total_step)

                vlosses.append(vloss)
                alosses.append(aloss)
                entropies.append(entropy)
                memory.after_update() # Need to Check!

            anum = args.red + args.blue
            print('Value loss : {:.2f} Action loss : {:.2f} Entropy : {:.2f} Utility : {:2f}'.format(\
                    sum(vlosses)/anum, sum(alosses)/anum, sum(entropies)/anum, sum(epi_rews)/len(epi_rews)))
            epi_rews = []
            if ep % args.save_freq == 0:
                hparams = AGENT_CONFIG[args.agent_type][1]
                base_fn = 'ep_{}_lr_{}_bs_{}_seq_{}.pt'.format(ep, hparams['lr'], args.batch_size, hparams['seq_len'])
                for i, agent in enumerate(agents):
                    agent_fn = 'agent{}_prefer_{}_'.format(i, prefer[i]) + base_fn
                    tr.save(agent.state_dict(), osp.join(args.save_dir,agent_fn))

if __name__ == '__main__':
    args = parse_args()
    main(args)
