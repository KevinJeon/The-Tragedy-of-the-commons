import names, random
import numpy as np

import components.world as world
from components.view import View

class Action:
    No_Op = 0

    Move_Up = 1
    Move_Down = 2
    Move_Left = 3
    Move_Right = 4

    Rotate_Left = 5
    Rotate_Right = 6

    Attack = 7


class Direction(object):
    def __init__(self):
        pass


class DirectionType:
    # Clock-wise numbering
    Up = 0
    Down = 2
    Left = 1
    Right = 4


class Direction(object):

    def __init__(self, direction_type):
        self.direction = direction_type

    def turn_right(self) -> Direction:
        self.direction = self.direction + 1 % 4
        return self

    def turn_left(self) -> Direction:
        self.direction = (self.direction + 4) - 1 % 4
        return self

    def half_rotate(self) -> Direction:
        self.direction = self.direction + 2 % 4
        return self

    def _to_string(self) -> str:
        if self.direction == DirectionType.Up:
            return 'Up'
        elif self.direction == DirectionType.Down:
            return 'Down'
        elif self.direction == DirectionType.Left:
            return 'Left'
        elif self.direction == DirectionType.Right:
            return 'Right'

    def __str__(self):
        return 'Direction({0})'.format(self._to_string())



class Agent(object):

    def __init__(self, world: world.World, pos: world.Position, name=None):
        self.world = world
        self.position = pos
        self.direction = Direction(direction_type=random.randint(1, 4))

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
        if direction is DirectionType.Up:
            new_pos = self.position + world.Position(x=0, y=1)
        elif direction is DirectionType.Down:
            new_pos = self.position + world.Position(x=0, y=-1)
        elif direction is DirectionType.Left:
            new_pos = self.position + world.Position(x=-1, y=0)
        elif direction is DirectionType.Right:
            new_pos = self.position + world.Position(x=1, y=0)

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

    def get_position(self) -> world.Position:
        return self.position

    def reset_reward(self) -> int:
        tick_reward = self.tick_reward
        self.tick_reward = 0.
        return tick_reward

    def get_view(self, output_type: str) -> np.array:
        assert output_type in ['rgb_array', 'numeric']

        positions = View.get_visible_positions(self.direction)

        for position in positions:
            pass

        raise NotImplementedError


    def __repr__(self):
        return '<Agent (name={0}, position={1}, direction={2})>'.format(self.name, self.position, self.direction)

# Lazy import (Circular import issue)
import components.item as items