import names

import components.world as world


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

    def move(self, pos: world.Position):
        raise NotImplementedError

    def attack(self, direction: Direction):
        raise NotImplementedError


