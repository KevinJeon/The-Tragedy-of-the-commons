from copy import deepcopy
import numpy as np
import random


class World(object):
    def __init__(self): pass


class Position(object):
    def __init__(self, x=None, y=None):
        assert x is not None
        assert y is not None
        self.x = x
        self.y = y

    def __mul__(self, scale: int):
        return Position(x=self.x * scale, y=self.y * scale)

    def __str__(self):
        return 'Position(x={0}, y={1})'.format(self.x, self.y)

    def to_tuple(self):
        return self.y , self.x


class Field(object):
    def __init__(self, world: World, p1: Position, p2: Position):
        self.world = world

        p1_x = p1.x if p1.x < p2.x else p1.x
        p1_y = p1.y if p1.y < p2.y else p2.y

        p2_x = p2.x if p1.x < p2.x else p1.x
        p2_y = p2.y if p2.y < p2.y else p2.y

        self.p1 = Position(x=p1_x, y=p1_y)
        self.p2 = Position(x=p2_x, y=p2_y)



class World(object):

    def __init__(self, size=(16, 16), num_agents=4):
        self.size = size
        self.agents = []
        self.grid = None
        self._build_grid()

        self.on_changed_callbacks = []
        self.fruits_fields = []

        self.num_agents = num_agents

        # TODO Hard-coded, but should change to random sampled
        self.add_fruits_field(Field(
                world=self,
                p1=Position(3, 10),
                p2=Position(7, 14),
            )
        )
        self.add_fruits_field(Field(
                world=self,
                p1=Position(0, 3),
                p2=Position(3, 6),
            )
        )
        self.add_fruits_field(Field(
                world=self,
                p1=Position(13, 15),
                p2=Position(14, 10),
            )
        )

        self._spawn_random_agents()



    def _build_grid(self):
        self.grid = np.empty(shape=(self.size), dtype=object)

    def _spawn_random_agents(self):
        for _ in range(self.num_agents):
            pos = Position(x=random.randint(0,  self.width - 1), y=random.randint(0, self.height - 1))
            self.spawn_agent(pos=pos)

    def spawn_agent(self, pos: Position):
        spawned = agent.Agent(world=self, pos=pos)
        self.agents.append(spawned)
        return spawned

    def spawn_block(self, pos: Position):
        spawned = block.Block(world=self)
        self.grid[pos.y][pos.x] = block
        return spawned

    def add_fruits_field(self, field: Field):
        self.fruits_fields.append(field)

    def get_agents(self) -> []:
        return self.agents

    @property
    def width(self) -> int:
        return self.size[1]

    @property
    def height(self) -> int:
        return self.size[0]



# Lazy import (Circular import issue)
import components.agent as agent
import components.block as block
