import names, random
import numpy as np
from copy import deepcopy

class Agent(object):
    def __init__(self):
        pass


import components.skill as skills
import components.world as world

from components.position import Position
from components.direction import DirectionType
from components.direction import Direction
from components.block import BlockType
import components.view as view



class Action:
    No_Op = 0

    Move_Up = 1
    Move_Down = 2
    Move_Left = 3
    Move_Right = 4

    Rotate_Right = 5
    Rotate_Left = 6

    Attack = 7


    @property
    def count(self):
        return 8



class Agent(object):

    def __init__(self, world: world.World, pos: Position, name=None):
        self.world = world
        self.position = pos
        self.direction = Direction(direction_type=random.randint(0, 3))

        self.name = name if name is not None else names.get_full_name()

        # Agent's accumulated reward during one step
        self.tick_reward = 0.

    def act(self, action: Action):
        if action is Action.Move_Up:
            self._move(DirectionType.Up)
        elif action is Action.Move_Down:
            self._move(DirectionType.Down)
        elif action is Action.Move_Left:
            self._move(DirectionType.Left)
        elif action is Action.Move_Right:
            self._move(DirectionType.Right)
        elif action is Action.No_Op:
            pass
        elif action is Action.Rotate_Left:
            self._rotate(DirectionType.Left)
        elif action is Action.Rotate_Right:
            self._rotate(DirectionType.Right)
        elif action is Action.Attack:
            self._attack()
        else:
            raise IndexError('Unknown action')

    def _move(self, direction: DirectionType):

        new_pos = None

        if direction == DirectionType.Up:
            if self.direction.value == DirectionType.Up:
                new_pos = self.position + Position(x=0, y=1)
            elif self.direction.value == DirectionType.Down:
                new_pos = self.position + Position(x=0, y=-1)
            elif self.direction.value == DirectionType.Left:
                new_pos = self.position + Position(x=-1, y=0)
            elif self.direction.value == DirectionType.Right:
                new_pos = self.position + Position(x=1, y=0)

        elif direction == DirectionType.Down:
            if self.direction.value == DirectionType.Up:
                new_pos = self.position - Position(x=0, y=1)
            elif self.direction.value == DirectionType.Down:
                new_pos = self.position + Position(x=0, y=1)
            elif self.direction.value == DirectionType.Left:
                new_pos = self.position + Position(x=1, y=0)
            elif self.direction.value == DirectionType.Right:
                new_pos = self.position - Position(x=1, y=0)

        elif direction == DirectionType.Left:
            if self.direction.value == DirectionType.Up:
                new_pos = self.position + Position(x=-1, y=0)
            elif self.direction.value == DirectionType.Down:
                new_pos = self.position + Position(x=1, y=0)
            elif self.direction.value == DirectionType.Left:
                new_pos = self.position + Position(x=0, y=-1)
            elif self.direction.value == DirectionType.Right:
                new_pos = self.position + Position(x=0, y=1)

        elif direction == DirectionType.Right:
            if self.direction.value == DirectionType.Up:
                new_pos = self.position + Position(x=1, y=0)
            elif self.direction.value == DirectionType.Down:
                new_pos = self.position - Position(x=1, y=0)
            elif self.direction.value == DirectionType.Left:
                new_pos = self.position + Position(x=0, y=1)
            elif self.direction.value == DirectionType.Right:
                new_pos = self.position - Position(x=0, y=1)

        if not self.world.get_agent(new_pos) is not None: # If other agent exists
            if self.world.map_contains(new_pos):
                self.position = new_pos

            self._try_gather()

        return self

    def _rotate(self, direction: DirectionType):
        if direction is DirectionType.Left:
            self.direction.turn_left()
        elif direction is DirectionType.Right:
            self.direction.turn_right()
        else:
            raise IndexError('Unknown direction type')

    def _attack(self) -> None:
        punish = skills.Punish()
        punish_positions = punish.get_targets(direction=self.direction)

        for position_row in punish_positions:

            for position in position_row:
                if position is None: continue

                self.world.apply_effect(self.position + position, punish)

        self.tick_reward += punish.reward
    def _try_gather(self):
        item = self.world.correct_item(pos=self.position)

        if isinstance(item, items.Apple):
            self.tick_reward += item.reward

        return self

    def get_position(self) -> Position:
        return self.position

    def reset_reward(self) -> int:
        tick_reward = self.tick_reward
        self.tick_reward = 0.
        return tick_reward

    def get_visible_positions(self, absolute=False) -> [Position]:
        '''
        :return: Agent's visible relative positions
        '''
        related_positions = view.View.get_visible_positions(self.direction)

        if absolute:
            for y, row in enumerate(related_positions):
                for x, item in enumerate(row):
                    related_positions[y][x] = item + self.position
        return related_positions

    def get_view(self) -> [BlockType]:
        positions = view.View.get_visible_positions(self.direction)
        positions = np.array(positions, dtype=object)

        # Fill agent on grid
        grid = deepcopy(self.world.grid)
        effects = deepcopy(self.world.effects)

        for iter_agent in self.world.agents:
            position = iter_agent.position
            grid[position.y][position.x] = iter_agent

        # Empty space to draw information
        sketch = np.zeros(positions.shape, dtype=np.uint64)

        for y, position_row in enumerate(positions):
            for x, position in enumerate(position_row):
                abs_position = position + self.position
                if self.world.map_contains(abs_position):  # If position is outside of map
                    item = grid[abs_position.y][abs_position.x]

                    if item is None:  # If item or agent exists on the position
                        sketch[y][x] = BlockType.Empty
                    else:
                        if isinstance(item, Agent):
                            if item == self:  # If the agent is myself
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.Self)
                            else:  # Or agent is companion or opponent
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.Others)
                        elif isinstance(item, items.Apple):
                            sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.Apple)

                    effect = effects[abs_position.y][abs_position.x]

                    if np.bitwise_and(int(effect), BlockType.Punish):
                        sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.Punish)

                else:
                    sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.OutBound)

        return sketch

    def get_view_as_type(self):
        positions = view.View.get_visible_positions(self.direction)
        positions = np.array(positions, dtype=object)

        # Fill agent on grid
        grid = deepcopy(self.world.grid)

        for iter_agent in self.world.agents:
            position = iter_agent.position
            grid[position.y][position.x] = iter_agent

        # Empty space to draw information
        sketch = np.zeros(positions.shape, dtype=np.uint64)

        for y, position_row in enumerate(positions):
            for x, position in enumerate(position_row):
                abs_position = position + self.position
                if self.world.map_contains(abs_position):  # If position is outside of map
                    item = grid[abs_position.y][abs_position.x]

                    if item is None:  # If item or agent exists on the position
                        sketch[y][x] = 0
                    else:
                        if isinstance(item, Agent):
                            if item == self:  # If the agent is myself
                                sketch[y][x] = 3
                            else:  # Or agent is companion or opponent
                                sketch[y][x] = 4
                        elif isinstance(item, items.Apple):
                            sketch[y][x] = 2

                else:
                    sketch[y][x] = 1

        return sketch

    def __repr__(self):
        return '<Agent (name={0}, position={1}, direction={2})>'.format(self.name, self.position, self.direction)


# Lazy import (Circular import issue)
import components.item as items