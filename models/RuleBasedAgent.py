import random
import cv2
from models.Agent import Agent

from components.observation import NumericObservation


class RuleBasedAgent(Agent):

    def __init__(self, color):
        super(RuleBasedAgent, self).__init__()

        self.color = color

    def act(self, obs):
        print(obs)
        cv2.imshow('Observation', obs)
        cv2.waitKey(0)

        # weight[]

        for y, row in enumerate(obs):
            for x, data in enumerate(row):
                print(x, y, data)
                # TODO Calculate distance from agent

        # NumericObservation.BlueApple



        return random.randint(0, 7)




