import cv2 as cv

from tocenv.env import TOCEnv
from pprint import pprint

from models.RuleBasedAgent import RuleBasedAgent


def main():

    agents_types = ['red', 'blue']

    env = TOCEnv(agents=agents_types,
                 map_size=(16, 16),
                 obs_type='numeric',
                 patch_count=8,
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
            if True in done: exit(1)
            print(info)
            image = env.render(coordination=True)

            state = next_state


if __name__ == '__main__':
    main()
