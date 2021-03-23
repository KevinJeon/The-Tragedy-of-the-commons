from components.world import Position
from components.agent import DirectionType


class View(object):

    @staticmethod
    def get_visible_positions(direction: DirectionType) -> None:
        if direction is DirectionType.Up:
            positions = [
                Position(-1, 3), Position(0, 3), Position(1, 3),
                Position(-1, 2), Position(0, 2), Position(1, 2),
                Position(-1, 1), Position(0, 1), Position(1, 1),
                Position(-1, 0), Position(0, 0), Position(1, 0),
                Position(-1, -1), Position(0, -1), Position(1, -1),
            ]
        elif direction is DirectionType.Down:
            positions = [
                Position(-1, 1), Position(0, 1), Position(1, 1),
                Position(-1, 0), Position(0, 0), Position(1, 0),
                Position(-1, -1), Position(0, -1), Position(1, -1),
                Position(-1, -2), Position(0, -2), Position(1, -2),
                Position(-1, -3), Position(0, -3), Position(1, -3),
            ]
        elif direction is DirectionType.Left:
            positions = [
                Position(-3, 1), Position(-2, 1), Position(-1, 1), Position(0, 1), Position(1, 1),
                Position(-3, 0), Position(-2, 0), Position(-1, 0), Position(0, 0), Position(1, 0),
                Position(-3, -1), Position(-2, -1), Position(-1, -1), Position(0, -1), Position(1, -1),
            ]
        elif direction is DirectionType.Right:
            positions = [
                Position(-1, 1), Position(0, 1), Position(1, 1), Position(2, 1), Position(3, 1),
                Position(-1, 0), Position(0, 0), Position(1, 0), Position(2, 0), Position(3, 0),
                Position(-1, -1), Position(0, -1), Position(1, -1), Position(2, -1), Position(3, -1),
            ]

        return positions
