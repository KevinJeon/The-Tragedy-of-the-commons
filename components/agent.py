import names, random
import numpy as np
from copy import deepcopy


import components.world as world
from components.block import BlockType
from components.position import Position


class DirectionType:
    # Clock-wise numbering
    Up = 0
    Down = 2
    Left = 1
    Right = 3


from components.view import View


class Action:
    No_Op = 0

    Move_Up = 1
    Move_Down = 2
    Move_Left = 3
    Move_Right = 4

    Rotate_Right = 5
    Rotate_Left = 6


    Attack = 7


class Direction(object):
    def __init__(self):
        pass


class Direction(object):

    def __init__(self, direction_type):
        self.direction = direction_type

    def turn_right(self) -> Direction:
        self.direction = (self.direction + 1) % 4
        return self

    def turn_left(self) -> Direction:
        self.direction = ((self.direction + 4) - 1) % 4
        return self

    def half_rotate(self) -> Direction:
        self.direction = (self.direction + 2) % 4
        return self

    @property
    def value(self):
        return self.direction

    def _to_position(self) -> Position:
        if self.direction == DirectionType.Up:
            return Position(x=0, y=1)
        elif self.direction == DirectionType.Down:
            return Position(x=0, y=-1)
        elif self.direction == DirectionType.Left:
            return Position(x=-1, y=0)
        elif self.direction == DirectionType.Right:
            return Position(x=1, y=0)

    def _to_string(self) -> str:
        if self.direction == DirectionType.Up:
            return 'Up'
        elif self.direction == DirectionType.Down:
            return 'Down'
        elif self.direction == DirectionType.Left:
            return 'Left'
        elif self.direction == DirectionType.Right:
            return 'Right'

    def get_type(self):
        return self.direction

    def __str__(self):
        return 'Direction({0})'.format(self._to_string())


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
        else:
            raise IndexError('Unknown action')

    def _move(self, direction: DirectionType):

        new_pos = None
        print(new_pos, direction, self.direction.value, DirectionType.Left)
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

    def _attack(self, direction: Direction):
        raise NotImplementedError

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
        related_positions = View.get_visible_positions(self.direction)

        if absolute:
            for y, row in enumerate(related_positions):
                for x, item in enumerate(row):
                    related_positions[y][x] = item + self.position
        return related_positions

    def get_view(self) -> [BlockType]:
        positions = View.get_visible_positions(self.direction)
        positions = np.array(positions, dtype=object)

        # Fill agent on grid
        grid = deepcopy(self.world.grid)
        for iter_agent in self.world.agents:
            position = iter_agent.position
            grid[position.y][position.x] = iter_agent

        # Empty space to draw information
        sketch = np.empty(positions.shape, dtype=np.int8)

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
                                sketch[y][x] = BlockType.Self
                            else:  # Or agent is companion or opponent
                                sketch[y][x] = BlockType.Others
                        elif isinstance(item, items.Apple):
                            sketch[y][x] = BlockType.Apple

                else:
                    sketch[y][x] = BlockType.OutBound

        return sketch

    def __repr__(self):
        return '<Agent (name={0}, position={1}, direction={2})>'.format(self.name, self.position, self.direction)

# Lazy import (Circular import issue)
import components.item as items