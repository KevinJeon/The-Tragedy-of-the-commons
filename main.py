import cv2 as cv
import random


from env import TOCEnv
from components.agent import Action


def main():
    num_agents = 4
    env = TOCEnv(num_agents=num_agents, map_size=(16, 16))

    while True:
        _ = env.reset()

        for i in range(100):

            image = env.render()
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
    main()