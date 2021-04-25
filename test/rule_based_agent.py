import cv2 as cv
import random
import numpy as np

from env import TOCEnv
from pprint import pprint
import cv2

from models.RuleBasedAgent import RuleBasedAgent


def main():

    agents_types = ['red']

    env = TOCEnv(agents=agents_types,
                 map_size=(16, 16),
                 obs_type='numeric',
                 apple_color_ratio=0.5,
                 apple_spawn_ratio=0.1,
                 )

    agents = [RuleBasedAgent(color=color) for color in agents_types]

    while True:
        state, info = env.reset()
        pprint(info)
        for i in range(400):

            image = env.render(coordination=True)
            cv.imshow('Env', image)
            key = cv.waitKey(1)

            actions = [agent.act(obs) for agent, obs in zip(agents, state)]

            next_state, reward, done, info = env.step(actions=actions)

            image = env.render(coordination=True)
            cv.imshow('Env', image)
            cv.waitKey(1)

if __name__ == '__main__':
    main()
