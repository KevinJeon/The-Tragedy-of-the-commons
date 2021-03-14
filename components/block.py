
class BlockType(object):
    Apple = 'Apple'
    Empty = 'Empty'


class Block(object):

    def __init__(self, world, type=BlockType.Empty) -> None:
        self.world = world
        self.type = type






