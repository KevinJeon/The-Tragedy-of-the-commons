import cv2 as cv
import random
import numpy as np
import ray

import env
from env import TOCEnv, ParallelTOCEnv
from pprint import pprint
import cv2

import logging
logging.basicConfig(level=logging.DEBUG)


def main():

    agents_types = ['red', 'blue', 'red']
    num_agents = len(agents_types)

    env = ParallelTOCEnv(
                num_envs=10,
                agents=agents_types,
                 map_size=(16, 16),
                 obs_type='rgb_array',
                 apple_color_ratio=0.5,
                 apple_spawn_ratio=0.1,
                 )

    while True:
        _ = env.reset()
        # pprint(info)
        for i in range(400):

            # image = env.render(coordination=False)
            # cv.imshow('Env', image)
            key = cv.waitKey(0)
            if key == 0: # Up
                action_1 = 1
            elif key == 1: # Down
                action_1 = 2
            elif key == 2: # Left
                action_1 = 3
            elif key == 3: # Right
                action_1 = 4
            elif key == 113: # Q
                action_1 = 5
            elif key == 119: # W
                action_1 = 6
            elif key == 97: # A
                action_1 = 7
            else: # No-op
                action_1 = 0
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
            if action_1 is not None:
                sampled_action.append(action_1)
                sampled_action.extend([random.randint(0, 7) for _ in range(num_agents - 1)])
            else:
                sampled_action = [random.randint(0, 7) for _ in range(num_agents)]

            _ = env.step(actions=sampled_action)


            # pprint(info)

            # image = env.render(coordination=False)
            # cv.imshow('Env', image)


if __name__ == '__main__':
    # result = ray.init(local_mode=True)
    import logging
    #result = ray.init(address='127.0.0.1:6379')

    # ray.util.connect('172.27.186.221:6379')


    result = ray.init(address='localhost:6379')

    print(result)
    main()
    ray.shutdown()
