from components.world import Position
from components.agent import DirectionType


class View(object):

    @staticmethod
    def get_visible_positions(direction: DirectionType) -> [Position]:
        direction = direction.get_type()

        if direction == DirectionType.Up:
            positions = [
                Position(x=-1, y=3), Position(x=0, y=3), Position(x=1, y=3),
                Position(x=-1, y=2), Position(x=0, y=2), Position(x=1, y=2),
                Position(x=-1, y=1), Position(x=0, y=1), Position(x=1, y=1),
                Position(x=-1, y=0), Position(x=0, y=0), Position(x=1, y=0),
                Position(x=-1, y=-1), Position(x=0, y=-1), Position(x=1, y=-1),
            ]
        elif direction == DirectionType.Down:
            positions = [
                Position(x=-1, y=1), Position(x=0, y=1), Position(x=1, y=1),
                Position(x=-1, y=0), Position(x=0, y=0), Position(x=1, y=0),
                Position(x=-1, y=-1), Position(x=0, y=-1), Position(x=1, y=-1),
                Position(x=-1, y=-2), Position(x=0, y=-2), Position(x=1, y=-2),
                Position(x=-1, y=-3), Position(x=0, y=-3), Position(x=1, y=-3),
            ]
        elif direction == DirectionType.Left:
            positions = [
                Position(x=-3, y=1), Position(x=-2, y=1), Position(x=-1, y=1), Position(x=0, y=1), Position(x=1, y=1),
                Position(x=-3, y=0), Position(x=-2, y=0), Position(x=-1, y=0), Position(x=0, y=0), Position(x=1, y=0),
                Position(x=-3, y=-1), Position(x=-2, y=-1), Position(x=-1, y=-1), Position(x=0, y=-1), Position(x=1, y=-1),
            ]
        elif direction == DirectionType.Right:
            positions = [
                Position(x=-1, y=1), Position(x=0, y=1), Position(x=1, y=1), Position(x=2, y=1), Position(x=3, y=1),
                Position(x=-1, y=0), Position(x=0, y=0), Position(x=1, y=0), Position(x=2, y=0), Position(x=3, y=0),
                Position(x=-1, y=-1), Position(x=0, y=-1), Position(x=1, y=-1), Position(x=2, y=-1), Position(x=3, y=-1),
            ]

        return positions
