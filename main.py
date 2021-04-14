import cv2 as cv
import random
import numpy as np

from env import TOCEnv

import cv2

def main():

    agents_types = ['red', 'blue', 'red']
    num_agents = len(agents_types)

    env = TOCEnv(agents=agents_types,
                 map_size=(16, 16),
                 obs_type='rgb_array',
                 apple_color_ratio=0.5,
                 apple_spawn_ratio=0.1,
                 )

    while True:
        _ = env.reset()

        for i in range(400):

            image = env.render(coordination=False)
            cv.imshow('Env', image)
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

            next_state, reward, done, info = env.step(actions=sampled_action)

            image = env.render(coordination=False)
            cv.imshow('Env', image)
            cv.waitKey(0)
            print(next_state.shape, reward, done, info)


if __name__ == '__main__':
    main()
