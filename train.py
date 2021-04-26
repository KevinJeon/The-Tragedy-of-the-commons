import cv2 as cv
import random
import numpy as np
import argparse
from env import TOCEnv
from models.a2c_toc import CPCAgent
from models.rulebased import RuleBasedAgent
from utils.storage import RolloutStorage

AGENT_TYPE = dict(rule=RuleBasedAgent, ac=CPCAgent)
AGENT_CONFIG = dict(rule=dict(prefer=None, obs_size=None), ac=dict(batch_size=128, seq_len=20, num_action=8, num_channel=3, timestep=100)) 

def parse_args():
    parser = argparse.ArgumentParser(description='ToC params')
    parser.add_argument('--blue', default=1, type=int)
    parser.add_argument('--red', default=1, type=int)
    parser.add_argument('--num_episode', default=3, type=int)
    parser.add_argument('--max_step', '-m', default=10, type=int)
    parser.add_argument('--agent_type', '-a', default='rule', type=str)
    parser.add_argument('--batch_size', '-bs', default=2, type=int)
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
    prefer = ['blue']*args.blue+['red']*args.red
    env = TOCEnv(agents=prefer, apple_color_ratio=0.5, apple_spawn_ratio=0.1)
    agents = [AGENT_TYPE[args.agent_type](**AGENT_CONFIG[args.agent_type]) for i in range(args.blue + args.red)]
    env.obs_type = 'rgb_array'
    memory = RolloutStorage(agent_type='ac',num_agent=args.blue+args.red, num_step=args.max_step, \
            batch_size=args.batch_size, num_obs=(88, 88, 3), num_action=8, num_rec=128)
    for ep in range(args.num_episode):
        obss, color_agents = env.reset()
        #key = cv.waitKey(0) to debug
        for step in range(args.max_step):
            image = env.render(coordination=True)
            cv.imshow('Env', image)
            key = cv.waitKey(1)
            sampled_action = []
            sampled_actions, infos = select_actions(obss, agents, step, memory.h, memory.n)
            obss_next, rews, masks, _ = env.step(actions=sampled_actions)
            memory.add(obss, sampled_actions, rews, masks, infos)
            obss = obss_next
        print('-'*20+'Train!'+'-'*20)
        memory.n += 1
        if memory.n % args.batch_size == 0: # if (memory.n != 0) and (memory.n % args.batch_size == 0):
            for i, agent in enumerate(agents):
                # check for return calculate
                obss, acts, rews, rets, masks = memory.obs[i], memory.act[i], memory.rew[i], memory.ret[i], memory.mask[i]
                samples = (obss, acts, rews, rets, masks)
                infos = None
                if args.agent_type == 'ac':
                    vs, logprobs, hs, s_fs, a_fs = memory.val[i], memory.logprob[i], memory.h[i], memory.s_feat[i], memory.a_feat[i]
                    infos = (vs, logprobs, hs, s_fs, a_fs)

                agent.train(samples, infos)
                memory.after_update()
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
