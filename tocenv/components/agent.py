class Agent(object):
    pass


class BlueAgent(Agent):
    pass


class RedAgent(Agent):
    pass


import names, random
import numpy as np
from copy import deepcopy

from tocenv.components.observation import NumericObservation


class Color:
    '''
    Red = (255, 0, 0)
    Orange = (200, 0, 0)
    Blue = (0, 0, 255)
    White = (255, 255, 255)
    Green = (0, 255, 0)
    '''

    Green = (0, 255, 0)
    Purple = (200, 200, 0)
    Orange = (200, 0, 0)
    Blue = (0, 0, 255)


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
    def action_count(self):
        return 8


import tocenv.components.skill as skills
import tocenv.components.world as world

from tocenv.components.position import Position
from tocenv.components.direction import DirectionType
from tocenv.components.direction import Direction
from tocenv.components.block import BlockType
import tocenv.components.view as view


class Agent(object):

    def __init__(self, world: world.World, pos: Position, name=None):
        self.color = None

        self.world = world
        self.position = pos
        self.direction = Direction(direction_type=random.randint(0, 3))

        self.name = name if name is not None else names.get_full_name()

        # Agent's accumulated reward during one step
        self._tick_reward = 0.
        self._tick_apple_eaten = None

        self._eaten_apple_count = 0

        self._tick_used_punishment = False
        self._tick_punished = False
        self._tick_prev_action = None

        self.accum_reward = 0.


    def act(self, action: Action):
        action = int(action)
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

        self._tick_prev_action = action

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

        if not self.world.get_agent(new_pos) is not None: # If other ra_agent exists
            if self.world.map_contains(new_pos):
                self.position = new_pos

            self._try_gather()

        self.world.env.increase_movement_count()

        return self

    def _rotate(self, direction: DirectionType):
        if direction is DirectionType.Left:
            self.direction.turn_left()
        elif direction is DirectionType.Right:
            self.direction.turn_right()
        else:
            raise IndexError('Unknown direction type')

        self.world.env.increase_rotate_count()

    def _attack(self) -> None:
        punish = skills.Punish()
        punish_positions = punish.get_targets(direction=self.direction)

        for position_row in punish_positions:

            for position in position_row:
                if position is None: continue

                self.world.apply_effect(self.position + position, punish)

        self._tick_reward += punish.reward
        self._tick_used_punishment = True
        self.world.env.increase_punishing_count()

    def _try_gather(self):
        item = self.world.correct_item(pos=self.position)

        if isinstance(item, items.Apple):
            self._tick_reward += item.reward
            self._tick_apple_eaten = 'apple'
            self._eaten_apple_count += 1

        return self

    def get_position(self) -> Position:
        return self.position

    def on_punished(self, damage: float) -> None:
        self._tick_reward += damage
        self._tick_punished = True
        self.world.env.increase_punished_count()

    def tick(self) -> None:
        self.accum_reward += self._tick_reward

        self._tick_reward = 0.
        self._tick_apple_eaten = None
        self._tick_used_punishment = False
        self._tick_punished = False
        self._tick_prev_action = None

    def get_reward(self) -> int:
        return self._tick_reward

    def get_apple_eaten(self) -> bool:
        return self._tick_apple_eaten

    def get_used_punishment(self) -> bool:
        return self._tick_used_punishment

    def get_punished(self) -> bool:
        return self._tick_punished

    def get_prev_action(self) -> int:
        return self._tick_prev_action

    def reset_accumulated_reward(self):
        self.accum_reward = 0.

    def reset_statistics(self) -> None:
        self._eaten_apple_count = 0

    def gather_info(self) -> dict():
        info = dict()

        info['color'] = self.color
        info['position'] = {'x': self.position.x, 'y': self.position.y}
        info['prev_action'] = self.get_prev_action()
        info['direction'] = self.direction._to_string()
        info['punished'] = self.get_punished()
        info['punishing'] = self.get_used_punishment()
        info['eaten'] = self.get_apple_eaten()
        info['accum_reward'] = self.accum_reward
        info['eaten_apples'] = self._eaten_apple_count

        return info

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

        # Fill ra_agent on grid
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

                    if item is None:  # If item or ra_agent exists on the position
                        sketch[y][x] = BlockType.Empty
                    else:

                        if isinstance(item, Agent):
                            if isinstance(item, BlueAgent):  # If the ra_agent is myself
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.BlueAgent)
                            elif isinstance(item, PurpleAgent):  # Or ra_agent is companion or opponent
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.PurpleAgent)
                            elif isinstance(item, GreenAgent):  # Or ra_agent is companion or opponent
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.GreenAgent)
                            elif isinstance(item, OrangeAgent):  # Or ra_agent is companion or opponent
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.OrangeAgent)
                        elif isinstance(item, items.Apple):
                            if isinstance(item, items.BlueApple):  # If the ra_agent is myself
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.BlueApple)
                            elif isinstance(item, items.RedApple):  # Or ra_agent is companion or opponent
                                sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.RedApple)

                    effect = effects[abs_position.y][abs_position.x]

                    if np.bitwise_and(int(effect), BlockType.Punish):
                        sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.Punish)

                else:
                    sketch[y][x] = np.bitwise_or(int(sketch[y][x]), BlockType.OutBound)

        return sketch

    def get_view_as_type(self):
        positions = view.View.get_visible_positions(self.direction)
        positions = np.array(positions, dtype=object)

        # Fill ra_agent on grid
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

                    if item is None:  # If item or ra_agent exists on the position
                        sketch[y][x] = NumericObservation.Empty
                    else:
                        if isinstance(item, Agent):
                            if item == self:  # If the ra_agent is myself
                                sketch[y][x] = NumericObservation.Self
                            elif isinstance(item, BlueAgent):
                                sketch[y][x] = NumericObservation.BlueAgent
                            elif isinstance(item, PurpleAgent):
                                sketch[y][x] = NumericObservation.PurpleAgent
                            elif isinstance(item, OrangeAgent):
                                sketch[y][x] = NumericObservation.OrangeAgent
                            elif isinstance(item, GreenAgent):
                                sketch[y][x] = NumericObservation.GreenAgent
                        elif isinstance(item, items.Apple):
                            if isinstance(item, items.BlueApple):
                                sketch[y][x] = NumericObservation.BlueApple
                            elif isinstance(item, items.RedApple):
                                sketch[y][x] = NumericObservation.RedApple
                else:
                    sketch[y][x] = NumericObservation.Wall

        return sketch

    def __repr__(self):
        return '<Agent (name={0}, position={1}, direction={2})>'.format(self.name, self.position, self.direction)


class PurpleAgent(Agent):
    def __init__(self, world, pos):
        super(PurpleAgent, self).__init__(world=world, pos=pos)
        self.color = 'purple'


class GreenAgent(Agent):
    def __init__(self, world, pos):
        super(GreenAgent, self).__init__(world=world, pos=pos)
        self.color = 'green'


class BlueAgent(Agent):
    def __init__(self, world, pos):
        super(BlueAgent, self).__init__(world=world, pos=pos)
        self.color = 'blue'


class OrangeAgent(Agent):
    def __init__(self, world, pos):
        super(OrangeAgent, self).__init__(world=world, pos=pos)
        self.color = 'orange'

# Lazy import (Circular import issue)
import tocenv.components.item as items