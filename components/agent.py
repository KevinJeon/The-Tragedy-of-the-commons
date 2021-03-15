import names

import components.world as world


class Action:
    Move_Up         = 0
    Move_Down       = 1
    Move_Left       = 2
    Move_Right      = 3
    Attack_Up       = 4
    Attack_Down     = 5
    Attack_Left     = 6
    Attack_Right    = 7


class Direction:
    Up = 'Up'
    Down = 'Down'
    Right = 'Right'
    Left = 'Left'


class Agent(object):

    def __init__(self, world: world.World, pos: world.Position, name=None):
        self.world = world
        self.position = pos
        self.name = name if name is not None else names.get_full_name()

        # Agent's accumulated reward during one step;
        self.tick_reward = 0.

    def act(self, action: Action):
        if action is Action.Move_Up:
            self._move(Direction.Up)
        elif action is Action.Move_Down:
            self._move(Direction.Down)
        elif action is Action.Move_Left:
            self._move(Direction.Left)
        elif action is Action.Move_Right:
            self._move(Direction.Right)
        elif action is Action.Attack_Up:
            self._attack(Direction.Up)
        elif action is Action.Attack_Down:
            self._attack(Direction.Down)
        elif action is Action.Attack_Left:
            self._attack(Direction.Left)
        elif action is Action.Attack_Right:
            self._attack(Direction.Right)
        else:
            raise IndexError('Unknown action')

    def _move(self, direction: Direction):
        if direction is Direction.Up:
            new_pos = self.position + world.Position(x=0, y=1)
        elif direction is Direction.Down:
            new_pos = self.position + world.Position(x=0, y=-1)
        elif direction is Direction.Left:
            new_pos = self.position + world.Position(x=-1, y=0)
        elif direction is Direction.Right:
            new_pos = self.position + world.Position(x=1, y=0)

        # TODO Check position is in the map
        if self.world.map_contains(new_pos):
            self.position = new_pos

    def _attack(self, direction: Direction):
        raise NotImplementedError

    def __repr__(self):
        return '<Agent (name={0}, position={1})>'.format(self.name, self.position)



