import components.world as world


class Block(object):

    def __init__(self, _world: world.World) -> None:
        self.world = world
        self.position = None
        self.can_walk = False


class Wall(Block):

    def __init__(self) -> None:
        super(Wall, self).__init__()

        self.can_walk = False

        raise NotImplementedError







