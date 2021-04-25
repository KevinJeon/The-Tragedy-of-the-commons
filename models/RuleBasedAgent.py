import random
import cv2
from models.Agent import Agent


class RuleBasedAgent(Agent):

    def __init__(self, color):
        super(RuleBasedAgent, self).__init__()

        self.color = color

    def act(self, obs):
        print(obs)
        cv2.imshow('Observation', obs)
        cv2.waitKey(0)





        return random.randint(0, 7)




