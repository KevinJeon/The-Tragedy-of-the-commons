from typing import Union

import numpy as np
import random
import math


class World(object):
    pass


import components.item as items
import components.agent as agent
from components.position import Position
import components.skill as skills
from components.block import BlockType
from components.agent import Color


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


class VariousAppleField(Field):
    def __init__(self, world: World, p1: Position, p2: Position, prob: float, ratio: float):
        super(VariousAppleField, self).__init__(world=world, p1=p1, p2=p2)
        '''
        ratio: Apple re-spawn ratio for 'BlueApple' and 'RedApple' (Set for BlueApple)
        '''
        self.prob = prob
        self.ratio = ratio

    def tick(self):
        self.generate_item(prob=self.prob)

    def generate_item(self, prob=0.3):

        empty_positions = self._get_empty_positions()
        agent_positions = [iter_agent.position for iter_agent in self.world.agents]

        apple_count = min(1, len(empty_positions))

        sampled_positions = random.sample(empty_positions, k=apple_count)

        apples = [items.BlueApple, items.RedApple]
        spawned_apples = random.choices(apples, weights=(self.ratio, 1-self.ratio), k=len(sampled_positions))

        for pos, item in zip(sampled_positions, spawned_apples):

            if random.random() < prob:
                self.world.spawn_item(item(), pos)

    def force_spawn_item(self, ratio=0.4):

        return


    def _get_empty_positions(self) -> [Position]:
        positions = []
        for y in range(self.p1.y, self.p2.y + 1):
            for x in range(self.p1.x, self.p2.x + 1):
                pos = Position(x, y)
                if self.world.get_item(pos) is None and \
                            self.world.get_agent(pos) is None:
                    positions.append(pos)

        return positions


class World(object):

    def __init__(self, size):
        self.size = size
        self.agents = []
        self.grid = None
        self.effects = None
        self._build_grid()

        self.on_changed_callbacks = []
        self.fruits_fields = []

        self._create_random_field()
        self._spawn_random_agents()
        self.clear_effect()

    def _build_grid(self):
        self.grid = np.empty(shape=self.size, dtype=object)

    def _spawn_random_agents(self):
        pass

    def _create_random_field(self):
        pass

    def spawn_agent(self, pos: Position, color):
        if color == Color.Red:
            spawned = agent.RedAgent(world=self, pos=pos)
        else:
            spawned = agent.BlueAgent(world=self, pos=pos)
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

    def apply_effect(self, pos: Position, effect: skills.Skill) -> bool:
        if self.map_contains(pos=pos):

            if isinstance(effect, skills.Punish):
                for iter_agents in self.agents:
                    if iter_agents.position == pos:
                        iter_agents.tick_reward += effect.damage

                self.effects[pos.y][pos.x] = np.bitwise_or(int(self.effects[pos.y][pos.x]), BlockType.Punish)
            return True
        else:
            return False

    def clear_effect(self):
        self.effects = np.zeros(shape=self.size, dtype=np.uint64)

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
import components.skill as skills