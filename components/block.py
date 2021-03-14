import components.world as world


class Block(object):

    def __init__(self, _world: world.World) -> None:
        self.world = world
        self.position = None
        self.can_walk = False


class AppleSpawner(Block):

    def __init__(self) -> None:
        super(AppleSpawner, self).__init__()

        self.can_walk = True




class Wall(Block):

    def __init__(self) -> None:
        super(Wall, self).__init__()

        self.can_walk = False

        raise NotImplementedError







