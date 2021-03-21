from typing import Union

import numpy as np
import random
import math

class World(object):
    pass


class Position(object):
    pass


import components.item as items
import components.agent as agent


class Position(object):
    def __init__(self, x=None, y=None):
        assert x is not None
        assert y is not None
        self.x = x
        self.y = y

    def subtract_y(self, scalar: int) -> Position:
        return Position(x=self.x, y=scalar - self.y)

    def __add__(self, pos: Position):
        return Position(x=self.x + pos.x, y=self.y + pos.y)

    def __sub__(self, pos: Position):
        return Position(x=self.x - pos.x, y=self.y - pos.y)

    def __mul__(self, scale: int):
        return Position(x=self.x * scale, y=self.y * scale)

    def __eq__(self, other: Position):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return 'Position(x={0}, y={1})'.format(self.x, self.y)

    def to_tuple(self, reverse=False):
        if reverse:
            return self.x, self.y
        else:
            return self.y, self.x

    def get_distance(self, pos: Position):
        return math.sqrt(math.pow(self.x - pos.x, 2) + math.pow(self.y - pos.y, 2))

    def get_surrounded(self, radius: int) -> [Position]:
        surrounded = []
        for _y in range(radius * -1, radius + 1):
            for _x in range(radius * -1, radius + 1):
                _position = self + Position(x=_x, y=_y)
                distance = self.get_distance(_position)

                if distance <= radius:
                    surrounded.append(_position)
        return surrounded


class Field(object):
    def __init__(self, world: World, p1: Position, p2: Position):
        self.world = world

        p1_x = p1.x if p1.x < p2.x else p1.x
        p1_y = p1.y if p1.y < p2.y else p2.y

        p2_x = p2.x if p1.x < p2.x else p1.x
        p2_y = p2.y if p2.y < p2.y else p2.y

        self.p1 = Position(x=p1_x, y=p1_y)
        self.p2 = Position(x=p2_x, y=p2_y)

    @property
    def area(self):
        return (self.p2.x - self.p1.x + 1) * (self.p2.y - self.p1.y + 1)

    @property
    def positions(self):
        positions = []
        for y in range(self.p1.y, self.p2.y + 1):
            for x in range(self.p1.x, self.p2.x + 1):
                positions.append(Position(x=x, y=y))
        return positions

    def tick(self):
        self.generate_item()

    def force_spawn_item(self, ratio=0.5):
        positions = self.positions
        num_samples = max(math.ceil(len(positions) * ratio), 1)

        sampled_position = random.sample(positions, num_samples)
        for pos in sampled_position:
            self.world.spawn_item(items.Apple(), Position(x=pos.x, y=pos.y))

    def generate_item(self, prob=0.5**4):
        for y in range(self.p1.y, self.p2.y + 1):
            for x in range(self.p1.x, self.p2.x + 1):

                surrounded_positions = self.world.get_surrounded_positions(pos=Position(x=x, y=y), radius=3)
                surrounded_items = self.world.get_surrounded_items(pos=Position(x=x, y=y), radius=3)
                apple_ratio = len(surrounded_items) / len(surrounded_positions) * prob

                if random.random() < apple_ratio:
                    self.world.spawn_item(items.Apple(), Position(x=x, y=y))


class World(object):

    def __init__(self, num_agents, size):
        self.size = size
        self.agents = []
        self.grid = None
        self._build_grid()

        self.on_changed_callbacks = []
        self.fruits_fields = []
        self.num_agents = num_agents

        self._create_random_field()
        self._spawn_random_agents()

    def _build_grid(self):
        self.grid = np.empty(shape=self.size, dtype=object)

    def _spawn_random_agents(self):
        for _ in range(self.num_agents):
            pos = Position(x=random.randint(0,  self.width - 1), y=random.randint(0, self.height - 1))
            self.spawn_agent(pos=pos)

    def _create_random_field(self):
        # TODO Hard-coded, but should change to random sampled
        self.add_fruits_field(Field(
                world=self,
                p1=Position(1, 1),
                p2=Position(4, 4),
            )
        )
        self.add_fruits_field(Field(
                world=self,
                p1=Position(5, 5),
                p2=Position(6, 6),
            )
        )
        self.add_fruits_field(Field(
                world=self,
                p1=Position(12, 12),
                p2=Position(15, 15),
            )
        )


    def spawn_agent(self, pos: Position):
        spawned = agent.Agent(world=self, pos=pos)
        self.agents.append(spawned)
        return spawned

    def spawn_block(self, pos: Position):
        spawned = block.Block(world=self)
        self.grid[pos.y][pos.x] = block
        return spawned


    def spawn_item(self, item: items.Item, pos: Position) -> bool:
        if not self.map_contains(pos): return False

        if self.grid[pos.y][pos.x] is None:
            self.grid[pos.y][pos.x] = item
            return True
        else:
            return False

    def add_fruits_field(self, field: Field):
        self.fruits_fields.append(field)
        field.force_spawn_item()

    def get_agents(self) -> []:
        return self.agents

    def map_contains(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def get_item(self, pos: Position) -> Union[items.Item, None]:
        return self.grid[pos.y][pos.x]

    def get_agent(self, pos: Position) -> Union[agent.Agent, None]:
        for iter_agent in self.agents:
            if iter_agent.position == pos:
                return iter_agent
        return None

    def remove_item(self, pos: Position) -> bool:
        if self.get_item(pos):
            self.grid[pos.y][pos.x] = None
            return True
        else:
            return False

    def correct_item(self, pos: Position) -> Union[items.Item, None]:
        item = self.get_item(pos)

        if item:
            self.remove_item(pos)
            return item
        else:
            return None

    def get_surrounded_positions(self, pos: Position, radius: int) -> [Position]:
        positions = pos.get_surrounded(radius=radius)
        surr_positions = []
        for position in positions:
            if self.map_contains(position):
                surr_positions.append(position)
        return surr_positions

    def get_surrounded_items(self, pos: Position, radius: int) -> [items.Item]:
        positions = self.get_surrounded_positions(pos=pos, radius=radius)

        items = []
        for position in positions:
            item = self.get_item(pos=position)
            if item is not None:
                items.append(item)
        return items

    def tick(self):
        [field.tick() for field in self.fruits_fields]




    @property
    def width(self) -> int:
        return self.size[1]

    @property
    def height(self) -> int:
        return self.size[0]


# Lazy import (Circular import issue)
import components.agent as agent
import components.block as block
import components.item as items