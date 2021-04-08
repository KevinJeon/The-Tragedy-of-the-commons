import numpy as np

class RuleBasedAgent(object):

    def __init__(self, red_apple, blue_apple, justice, obs_size):
        # agent characteristics
        self.u = [red_apple, blue_apple]
        self.justice = justice # bool type to test inequity aversion
        
        # constant
        self.a = ((obs_size[0] + 1)/2 , 1)
        self.obs_size = obs_size
        self.movable = [(self.a[0] + 1, self.a[1]), (self.a[0] - 1, self.a[1]), (self.a[0], self.a[1] + 1), self.a[0], self.a[1] -1)] 
        self.rule_order = [self._eat_apple, self._explore] 
    def act(self, obs):
        # Each step, agent find the most closest apple. 
        
        # if can get apple
        act  = None
        o = 0
        while act == None:
            act = self.rule_order[o](obs)
            o += 1
        return act

    def _move_most_apple(self, obs):
        # get the index of apples 
        ## to develop to output action
        apples = np.where(obs == 2)
        return apples

    def _closest_apple(self, obs):
        # get the index of closest apple
        apples = np.where(obs == 2)


    def _explore(self, direction):
        # if there is no apple in sight, explore
        horizontal, vertical = [0, self.obs_size[0] - 1], [self.obs_size[1] - 1, 0] # Left, Right, Up, Down
        ## check plz obs indices mean x and y each
        # Check whether wall or not
        possible = []
        for i, (ud, zr) in enumerate(zip(vertical, horizontal)):
            if not all(obs[ud, :]) == 1:
                possible.append(i)
            if not all(obs[:, lr]) == 1:
                possible.append(i + 2)
        act = np.random.choice(possible, 1)[0]
        return act
        
        
    def _eat_apple(self, obs):
        candidates = []
        utilities = []
        for move in self._movable:
            is_apple = obs[move[0], move[1]] == 2
            if is_apple:
                # considering red, blue
                candidates.append(move)
                utilites.append(self.u[obs[move[0], move[1] - 2]]) # for now, -2 but update to fit with env
            if not len(candidates) == 0:
                return np.argmax(utilites)
            else:
                return None

