class Skill(object):
    def __init__(self):
        pass


from components.position import Position
from components.direction import DirectionType


class Punish(Skill):

    def __init__(self):
        self.reward = -1.
        self.damage = -50.

    def get_targets(self, direction: DirectionType) -> [Position]:
        if direction.value == DirectionType.Up:
            return [
                [None, Position(x=0, y=3), None],
                [Position(x=-1, y=2), Position(x=0, y=2), Position(x=1, y=2)],
                [Position(x=-1, y=1), Position(x=0, y=1), Position(x=1, y=1)],
                [Position(x=-1, y=0), None, Position(x=1, y=0)],
            ]

        elif direction.value == DirectionType.Down:
            return [
                [Position(x=-1, y=0), None, Position(x=1, y=0)],
                [Position(x=-1, y=-2), Position(x=0, y=-2), Position(x=1, y=-2)],
                [Position(x=-1, y=-1), Position(x=0, y=-1), Position(x=1, y=- 1)],
                [None, Position(x=0, y=-3), None],
            ]

        elif direction.value == DirectionType.Left:
            return [
                [None, Position(x=-2, y=1), Position(x=-1, y=1), Position(x=0, y=1)],
                [Position(x=-3, y=0), Position(x=-2, y=0), Position(x=-1, y=0), None],
                [None, Position(x=-2, y=-1), Position(x=-1, y=-1), Position(x=0, y=-1)],
            ]

        elif direction.value == DirectionType.Right:
            return [
                [Position(x=0, y=1), Position(x=1, y=1), Position(x=2, y=1), None],
                [None, Position(x=1, y=0), Position(x=2, y=0), Position(x=3, y=0)],
                [Position(x=0, y=-1), Position(x=1, y=-1), Position(x=2, y=-1), None],
            ]
