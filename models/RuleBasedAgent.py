import random
import cv2
import math
import numpy as np

from components.agent import Action
from models.Agent import Agent
from components.observation import NumericObservation


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


class RuleBasedAgent(Agent):

    def __init__(self, color):
        super(RuleBasedAgent, self).__init__()

        self.color = color
        self.agent_pos = (9, 5)

        ''' Settings '''
        self.main_favorable = 3
        self.sub_favorable = 1

    def act(self, obs):


        weight = self.get_weight_matrix(obs=obs)
        score = self.get_direction_score(weight=weight)

        score_softmax = softmax(score)
        choice = random.choices([Action.Move_Up, Action.Move_Down, Action.Move_Left, Action.Move_Right],
                                weights=score_softmax, k=1)

        if abs(np.sum(score)) < 0.05:
            choice = random.choices([Action.Rotate_Right, Action.Rotate_Left, Action.Move_Down], weights=[0.5, 0.5, 0.3], k=1)

        return choice[0]

    def get_weight_matrix(self, obs) -> np.array:
        weight = np.zeros(shape=np.array(obs).shape, dtype=np.float32)

        for y, row in enumerate(obs):
            for x, data in enumerate(row):
                dist = abs(self.agent_pos[0] - y) + abs(self.agent_pos[1] - x)

                if data in [NumericObservation.BlueApple, NumericObservation.RedApple]:
                    weight[y][x] += self.get_favorable(data) * (0.5 ** dist)

        return weight

    def get_direction_score(self, weight) -> np.array:
        score = np.zeros(shape=4, dtype=np.float32)
        # [UP, DOWN, LEFT, RIGHT]

        for y, row in enumerate(weight):
            for x, data in enumerate(row):
                if y < self.agent_pos[0]:
                    score[0] += data
                if y > self.agent_pos[0]:
                    score[1] += data

                if x < self.agent_pos[1]:
                    score[2] = data
                if x > self.agent_pos[1]:
                    score[3] += data

        return score

    def get_favorable(self, item):
        if self.color == 'red':
            if item == NumericObservation.RedApple:
                return self.main_favorable
            else:
                return self.sub_favorable
        elif self.color == 'blue':
            if item == NumericObservation.BlueApple:
                return self.main_favorable
            else:
                return self.sub_favorable
