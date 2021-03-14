import components.world as world
import names

import components.world as world

class Agent(object):
    def __init__(self, world: world.World, pos: world.Position, name=None):
        self.world = world
        self.position = pos
        self.name = name if name is not None else names.get_full_name()
