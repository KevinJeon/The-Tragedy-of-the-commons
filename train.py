import cv2 as cv
import argparse
from tocenv.env import TOCEnv
from models.a2c_toc import CPCAgent, CPCTrainer
from models.rulebased import RuleBasedAgent
from tocenv.utils.storage import RolloutStorage

import torch as tr
from torch.utils.tensorboard import SummaryWriter


AGENT_TYPE = dict(rule=[RuleBasedAgent, None], ac=[CPCAgent, CPCTrainer])
AGENT_CONFIG = dict(rule=[dict(prefer=None, obs_size=None), None],
        ac=[dict(batch_size=None, seq_len=20, num_action=8, num_channel=3, timestep=100), \
                dict(eps=1e-5, alpha=0.99, lr=5e-4, max_grad_norm=0.5, vloss_coef=0.5, entropy_coef=0.01)]) 

def parse_args():
    parser = argparse.ArgumentParser(description='ToC params')
    parser.add_argument('--blue', default=1, type=int)
    parser.add_argument('--red', default=1, type=int)
    parser.add_argument('--num_episode', default=3, type=int)
    parser.add_argument('--max_step', '-m', default=10, type=int)
    parser.add_argument('--agent_type', '-a', default='rule', type=str)
    parser.add_argument('--batch_size', '-bs', default=2, type=int)
    parser.add_argument('--tensorboard', action="store_true")
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

    if args.tensorboard:
        writer = SummaryWriter()
    else:
        writer = None

    prefer = ['blue']*args.blue+['red']*args.red
    env = TOCEnv(agents=prefer, apple_color_ratio=0.5, apple_spawn_ratio=0.1)
    if args.agent_type == 'ac':
        AGENT_CONFIG[args.agent_type][0]['batch_size'] = args.batch_size
    agents = [AGENT_TYPE[args.agent_type][0](**AGENT_CONFIG[args.agent_type][0]) for i in range(args.blue + args.red)]
    trainers = [AGENT_TYPE[args.agent_type][1](agent=agents[i], **AGENT_CONFIG[args.agent_type][1]) for i in range(args.blue + args.red)]
    env.obs_type = 'rgb_array'
    memory = RolloutStorage(agent_type='ac',num_agent=args.blue+args.red, num_step=args.max_step, \
            batch_size=args.batch_size, num_obs=(88, 88, 3), num_action=8, num_rec=128)

    total_step = 0

    for ep in range(args.num_episode):
        obss, color_agents = env.reset()
        #key = cv.waitKey(0) to debug
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

        ''' Log Environment Statistics '''
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
            print('Value loss : {:.2f} Action loss : {:.2f} Entropy : {:.2f}'.format(sum(vlosses)/anum, sum(alosses)/anum, sum(entropies)/anum))
if __name__ == '__main__':
    args = parse_args()
    main(args)
