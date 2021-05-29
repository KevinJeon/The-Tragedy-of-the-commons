class World(object):
    pass


class Field(object):
    pass


from typing import Union

import numpy as np
import random
import math

from tocenv.env import TOCEnv

import tocenv.components.item as items
import tocenv.components.agent as agent
import tocenv.components.skill as skills
import tocenv.components.block as block
from tocenv.components.position import Position
from tocenv.components.block import BlockType
from tocenv.components.agent import Color

from tocenv.components.util.weighted_random import get_weighted_position


class Field(object):
    def __init__(self, world: World, p1: Position, p2: Position):
        self.world = world

        p1_x = p1.x if p1.x < p2.x else p2.x
        p1_y = p1.y if p1.y < p2.y else p2.y

        p2_x = p2.x if p1.x < p2.x else p1.x
        p2_y = p2.y if p1.y < p2.y else p1.y

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

    @property
    def include(self, pos: Position):
        return (self.p1.x <= pos.x <= self.p2.x) and \
               (self.p1.y <= pos.y <= self.p2.y)

    @property
    def center(self):
        center_x = (self.p1.x + self.p2.x) // 2
        center_y = (self.p1.y + self.p2.y) // 2
        return Position(x=center_x, y=center_y)

    def is_overlap(self, field: Field):

        if self.p2.y < field.p1.y or self.p1.y > field.p2.y:
            return False

        if self.p2.x < field.p1.x or self.p1.x > field.p2.x:
            return False

        return True

    @staticmethod
    def create_from_parameter(world: World, pos: Position, radius: int):
        p1 = Position(pos.x - radius, pos.y - radius)
        p2 = Position(pos.x + radius, pos.y + radius)
        return Field(world=world, p1=p1, p2=p2)

    def tick(self):
        self.generate_item()

    def force_spawn_item(self, ratio=0.5):
        print(ratio)
        positions = self.positions
        num_samples = max(math.ceil(len(positions) * ratio), 1)

        sampled_position = random.sample(positions, num_samples)
        for pos in sampled_position:
            self.world.spawn_item(items.Apple(), Position(x=pos.x, y=pos.y))

    def generate_item(self, prob=0.5 ** 4):
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

    def force_spawn_item(self, ratio=0.5):

        positions = self.positions
        num_samples = max(math.ceil(len(positions) * ratio), 1)

        sampled_position = random.sample(positions, num_samples)

        apples = [items.BlueApple, items.RedApple]
        spawned_apples = random.choices(apples, weights=(self.ratio, 1-self.ratio), k=len(sampled_position))

        for pos, item in zip(sampled_position, spawned_apples):
            if random.random() < ratio:
                self.world.spawn_item(item(), pos)

    def _get_empty_positions(self) -> [Position]:
        positions = []
        for y in range(self.p1.y, self.p2.y + 1):
            for x in range(self.p1.x, self.p2.x + 1):
                pos = Position(x, y)
                if self.world.get_item(pos) is None and \
                            self.world.get_agent(pos) is None:
                    positions.append(pos)

        return positions

    @staticmethod
    def create_from_parameter(world: World, pos: Position, radius: int, prob: float, ratio: float):
        p1 = Position(pos.x - radius, pos.y - radius)
        p2 = Position(pos.x + radius, pos.y + radius)
        return VariousAppleField(world=world, p1=p1, p2=p2, prob=prob, ratio=ratio)


class World(object):

    def __init__(self,
                 env: TOCEnv,
                 size: tuple,
                 apple_color_ratio: float,
                 apple_spawn_ratio: float,
                 patch_count: int,
                 patch_distance: int):
        self.env = env
        self.size = size
        self.agents = []
        self.grid = None
        self.effects = None
        self._build_grid()

        self.on_changed_callbacks = []
        self.fruits_fields = []

        self.patch_count = patch_count
        self.patch_distance = patch_distance

        self.apple_color_ratio = apple_color_ratio
        self.apple_spawn_ratio = apple_spawn_ratio

        self._create_random_field()

        self.clear_effect()

    def _build_grid(self):
        self.grid = np.empty(shape=self.size, dtype=object)

            
    def _create_random_field(self):

        patch_size = 3
        half_size = patch_size // 2
        distance = self.patch_distance

        initial_pos = get_weighted_position(mu=0, sigma=1, map_size=self.size)
        initial_pos = Position(
            x=max(half_size + 1, min(initial_pos.x, self.width - half_size - 1)),
            y=max(half_size + 1, min(initial_pos.y, self.height - half_size - 1))
        )

        self.add_fruits_field(VariousAppleField.create_from_parameter(world=self, pos=initial_pos, radius=half_size, prob=self.apple_spawn_ratio, ratio=self.apple_color_ratio))

        bfs = BFS(world=self)
        searched_positions = bfs.search(pos=initial_pos, radius=half_size, distance=distance, \
                                        k=self.patch_count - 1)

        for pos in searched_positions:
            self.add_fruits_field(VariousAppleField.create_from_parameter(world=self, pos=pos, radius=half_size, prob=self.apple_spawn_ratio, ratio=self.apple_color_ratio))

    def spawn_agent(self, pos: Position, color: Color):
        if color == Color.Red:
            spawned = agent.RedAgent(world=self, pos=pos)
        elif color == Color.Blue:
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

    def contains_field(self, field: Field) -> bool:
        return self.map_contains(field.p1) and self.map_contains(field.p2)

    def collapsed_field_exist(self, field: Field) -> bool:
        for iter_field in self.fruits_fields:
            if field.is_overlap(iter_field): return True

        return False

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
                        iter_agents.on_punished(effect.damage)

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


from tocenv.components.algorithm.BFS import BFS
