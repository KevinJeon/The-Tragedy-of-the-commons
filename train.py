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

    env = TOCEnv(num_agents=args.num_agent, map_size=(16, 16))
    for ep in range(args.num_episode): 
        obss, _ = env.reset()
        agents = [CPCAgent() for _ range(args.num_agent)]
        rollout = Stoarge(args.max_step, args.batch_size, env.observaion_space.shape, env.action_space.n)
        for step in range(args.max_step):
            # image = env.render()
        
            # cv.imshow('Env', image)
            # key = cv.waitKeyEx()
            # key = cv.waitKey(1)
            actions, vs, logprobs, hs, s_fs, a_fs = select_actions(obss, agents)
            obss_next, rews, dones, infos = env.step(actions=actions)
            rollout.add(obss, obss_next, hs, actions, logprobs, vs, rews, dones)

            #image = env.render()
            #cv.imshow('Env', image)
            
            print(next_state.shape, reward, done, info)


if __name__ == '__main__':
    args = parse_args()
    main(args)
