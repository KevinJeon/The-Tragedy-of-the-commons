import cv2 as cv
import random
import argparse
from env import TOCEnv
from components.agent import Action
import config as cfgs
def parse_args():
    parser = argparse.ArgumenParser('TOC Params')
    parser.add_argument('--num_episode', '-e', default=10000, type=int)
    parser.add_argument('--batch_size', '-b', default=1024, type=int)
    parser.add_argument('--num_agent', '-a', default=5, type=int)
    parser.add_argument('--max_step', '-ms', default=500, type=int)
    parser.add_argument('--save_dir', '-s', default='/save', type=str)
    parser.add_argument('--log_dir', '-l', default='/logs', type=str)
    return parser.parse_args()
def main(args, cfgs):
    ecfgs, acfgs, mcfgs = cfgs.env_info, cfgs.agent_info, cfgs.model_info 
    # In args, args are emergent params that needs to change easily, otherwise, in cfgs

    env = TOCEnv(num_agents=num_agents, map_size=(16, 16))
    while True:
        _ = env.reset()

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
    main(args, cfgs)
