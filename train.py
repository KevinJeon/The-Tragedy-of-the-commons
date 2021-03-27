import cv2 as cv
import random
import numpy as np
import argparse
from env import TOCEnv
from models.a2c_toc import CPCAgent

def parse_args():
    parser = argparse.ArgumentParser(desc='ToC params')
    parser.add_argument('--num_agent', default=4, type=int)
    parser.add_argument('--num_episode', deafult=10000, type=int)
    parser.add_argument('--max_step', default=100, type=int)
    args = parse_args()
    return args
def select_actions(obss, agents):
    actions = []
    vs = []
    logprobs = []
    hs = []
    s_fs = []
    a_fs = []
    for obs, agent, h0 in zip(obss, agents, hs):
        v, act, logprob, h, s_f, a_f = agent.act(obs, h0)
        act = act.detach().numpy()
        action = np.argmax(act)
        actions.append(action)
        vs.append(v)
        logprobs.append(logprob)
        hs.append(h0)
        s_fs.append(s_f)
        a_fs.append(a_f)
    return actions, vs, logprobs, hs, s_fs, a_fs

def main(args):
    num_agents = 4


    env = TOCEnv(num_agents=num_agents, map_size=(16, 16))

    while True:
        obs = env.reset()

        for i in range(100):

            image = env.render()
            cv.imshow('Env', image)
            key = cv.waitKeyEx()

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

            sampled_action = []
            if action_1 is not None:
                sampled_action.append(action_1)
                sampled_action.extend([random.randint(0, 4) for _ in range(num_agents - 1)])
            else:
                sampled_action = [random.randint(0, 4) for _ in range(num_agents)]

            ret = env.step(actions=sampled_action)
            print(ret)



if __name__ == '__main__':
    args = parse_args()
    main(args)
