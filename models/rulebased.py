import numpy as np

class RuleBasedAgent:

    def __init__(self, red_apple, blue_apple, justice, obs_size):
        # agent characteristics
        self.u = [red_apple, blue_apple]
        self.justice = justice # bool type to test inequity aversion
        
        # constant
        self.a = (int((obs_size[0] + 1)/2) , 1)
        self.obs_size = obs_size
        self.movable = [(self.a[0], self.a[1] + 1), (self.a[0], self.a[1]- 1), (self.a[0] - 1, self.a[1]), (self.a[0] + 1, self.a[1])] 
        # Up, Down, Left, Right
        self.rule_order = [self._eat_apple, self._closest_apple, self._explore] 
        self.color = dict(red=3, blue=2)
        self.is_train = False
    def act(self, obs):
        # Each step, agent find the most closest apple. 
        
        # if can get apple
        act  = None
        o = 0
        while act == None:
            act = self.rule_order[o](obs)
            o += 1
        return act, self.rule_order[o]

    def _move_most_apple(self, obs, color='red'):
        # get the index of apples 
        ## to develop to output action
        apples = np.where(obs == self.color[color])
        return apples

    def _closest_apple(self, obs, color='red'):
        # get the index of closest apple
        print(obs)
        apples = np.where(obs == self.color[color])
        xs, ys = apples[0], apples[1]
        closest = None
        lowest = 1e5
        if len(xs) == 0:
            return None
        for x, y in zip(xs, ys):
            arr = np.array([x,y])
            dist = np.linalg.norm(arr - self.a)
            if dist < lowest:
                lowest = dist
                closest = apple
        
        lr, ud = self.a - closest
        candidates = []
        if lr < 0:
            candidates.append(3)
        elif lr == 0:
            pass
        else:
            candidates.append(4)
        if ud < 0:
            candidates.append(2)
        elif ud == 0:
            pass
        else:
            candidates.append(1)
        return np.random.choice(candidates, 1)[0]

    def _explore(self, obs, color=None):
        # if there is no apple in sight, explore
        horizontal, vertical = [0, self.obs_size[0] - 1], [self.obs_size[1] - 1, 0] # Left, Right, Up, Down
        ## check plz obs indices mean x and y each
        # Check whether wall or not
        possible = []
        for i, (ud, lr) in enumerate(zip(vertical, horizontal)):
            if not np.mean(obs[ud, :]) == 1:
                possible.append(i)
            if not np.mean(obs[:, lr]) == 1:
                possible.append(i + 2)
        act = np.random.choice(possible, 1)[0]
        return act
        
        
    def _eat_apple(self, obs, color='red'):
        candidates = []
        utilities = []
        for i, move in enumerate(self.movable):
            is_apple = obs[move[0], move[1]] in [2, 3] 
            if is_apple:
                # considering red, blue
                candidates.append(i + 1)
                utilites.append(self.u[obs[move[0], move[1] - 2]]) # for now, -2 but update to fit with env e.g. red 2 blue 3  red_u 0 blue_u 1
            if not len(candidates) == 0:
                return np.argmax(utilites)
            else:
                return None

