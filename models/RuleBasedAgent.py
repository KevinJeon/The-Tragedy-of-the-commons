import random
import cv2
import math
import numpy as np

from models.Agent import Agent
from components.observation import NumericObservation


class RuleBasedAgent(Agent):

    def __init__(self, color):
        super(RuleBasedAgent, self).__init__()

        self.color = color
        self.agent_pos = (9, 5)


    def act(self, obs):
        print(obs)
        cv2.imshow('Observation', obs)
        cv2.waitKey(0)

        weight = np.zeros(shape=np.array(obs).shape, dtype=np.float32)

        for y, row in enumerate(obs):
            for x, data in enumerate(row):
                dist = abs(self.agent_pos[0] - y) + abs(self.agent_pos[1] - x)

                if data == NumericObservation.BlueApple:
                    weight[y][x] += 3.0 * ((0.95) ** dist)
                elif data == NumericObservation.RedApple:
                    weight[y][x] += 1.0 * ((0.95) ** dist)



        print(weight)
        # NumericObservation.BlueApple



        return random.randint(0, 7)




