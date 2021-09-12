import random
import numpy as np

from tocenv.components.agent import Action
from models.Agent import Agent
from tocenv.components.observation import NumericObservation


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


class RuleBasedAgent(Agent):

    def __init__(self, agent_type):
        super(RuleBasedAgent, self).__init__()

        self.color = agent_type
        self.agent_pos = (9, 5)

        ''' Settings '''
        self.main_favorable = 3
        self.sub_favorable = 1

    def act(self, obs) -> np.array:
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
                    weight[y][x] += self.main_favorable * (0.5 ** dist)

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


class RuleBasedAgentGroup(object):

    def __init__(self,
                 name,
                 agent_types,
                 obs_dim,
                 action_dim,
                 device,
                 batch_size):
        super(RuleBasedAgentGroup, self).__init__()

        self.name = name
        self.batch_size = batch_size
        self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.agent_types = agent_types
        self.agents = [RuleBasedAgent(
                 agent_type,
                ) for agent_type in self.agent_types]

    def act(self, obses, sample=False):
        joint_action = []

        for iter_obs, agent in zip(obses, self.agents):
            joint_action.append(agent.act(iter_obs))

        return np.array(joint_action)

    def train(self, memory, total_step, logger=None):
        pass
