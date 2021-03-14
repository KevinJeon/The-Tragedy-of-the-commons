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
        self.tick_reward = None

    def act(self, action: Action):
        pass

    def _move(self, direction: Direction):
        raise NotImplementedError

    def _attack(self, direction: Direction):
        raise NotImplementedError

    def __repr__(self):
        return '<Agent (name={0}, position={1})>'.format(self.name, self.position)



