from components.agent import Agent
# from components.block import Block
from components.world import World

import numpy as np


class TOCEnv(object):

    def __init__(self,
                 render=False
                 ):
        self.world = World()

        pass

    def step(self, actions: np.array):
        assert actions.shape is (self.world.num_agents, 1)

        for agent, action in zip(self.world.agents, actions):
            agent.act(action)

    def reset(self):
        pass

    def render(self):
        pass

    def get_full_state(self):
        return self.world.grid
