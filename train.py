import cv2 as cv
import random
import numpy as np
import argparse
from env import TOCEnv
from models.a2c_toc import CPCAgent
from models.rulebased import RuleBasedAgent

AGENT_TYPE = dict(rule=RuleBasedAgent, ac=CPCAgent)

def parse_args():
    parser = argparse.ArgumentParser(description='ToC params')
    parser.add_argument('--num_agent', default=10, type=int)
    parser.add_argument('--num_episode', default=10000, type=int)
    parser.add_argument('--max_step', default=100, type=int)
    parser.add_argument('--agent_type', default='rule', type=str)
    args = parser.parse_args()
    return args

def select_actions(obss, agents):
    actions = []
    infos = []
    for obs, agent in zip(obss, agents):
        act, info = agent.act(obs)
        actions.append(act)
        infos.append(info)
    return actions, infos

def main(args):
    env = TOCEnv(num_agents=args.num_agent, blue_agents=2, red_agents=1)
    prefer = ['blue']*50+['red']*50
    agents = [AGENT_TYPE[args.agent_type](prefer[i], False, (11, 11)) for i in range(args.num_agent)]
    env.obs_type = 'numeric'
    for ep in range(args.num_episode):
        obss, color_agents = env.reset()
        for i in range(args.max_step):
            image = env.render(coordination=True)
            cv.imshow('Env', image)
            key = cv.waitKey(1)
            '''
            if key == 0x260000: # Up
                action_1 = 1
            elif key == 0x280000: # Down
                action_1 = 2
            elif key == 0x250000: # Left
                action_1 = 3
            elif key == 0x270000: # Right
                action_1 = 4
            else: # No-op
                action_1 = None
            '''
            sampled_action = []
            sampled_actions, infos = select_actions(obss, agents)
            obss, rews, done, _ = env.step(actions=sampled_actions)
        '''
            memory.add(sampled_actions, infos)
        if memory.train:
            for agent in agents:
                if agent.is_train:
                    agent.train()
        '''

if __name__ == '__main__':
    args = parse_args()
    main(args)
