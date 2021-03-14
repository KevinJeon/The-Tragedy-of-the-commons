from copy import deepcopy


class Position(object):
    def __init__(self, x=None, y=None):
        assert x is not None
        assert y is not None
        self.x = x
        self.y = y


class World(object):

    def __init__(self, size=(32, 32), num_agents=4):
        self.size = size
        self.agents = []
        self.grid = None
        self._build_grid()

        self.on_changed_callbacks = []

        [self.spawn_agent(pos=Position(x=1, y=2)) for _ in range(num_agents)]

    def _build_grid(self):
        self.grid = None

    def spawn_agent(self, pos: Position):
        spawned = agent.Agent(world=self, pos=pos)
        self.agents.append(agent)
        return spawned

    def spawn_block(self, pos: Position):
        spawned = block.Block(world=self)
        self.grid[pos.y][pos.x] = block
        return spawned

    def get_agents(self) -> []:
        return deepcopy(self.agents)

    @property
    def width(self) -> int:
        return self.size(1)

    @property
    def height(self) -> int:
        return self.size(0)

    @property
    def num_agents(self) -> int:
        return len(self.agents)


# Lazy import (Circular import issue)
import components.agent as agent
import components.block as block
