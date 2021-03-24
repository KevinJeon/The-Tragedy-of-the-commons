from components.position import Position
from components.agent import DirectionType


class View(object):

    @staticmethod
    def get_visible_positions(direction: DirectionType) -> [Position]:
        direction = direction.get_type()

        positions = []
        if direction == DirectionType.Up:

            for x in range(-5, 6):
                _positions = []
                for y in range(9, -2, -1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Down:

            for x in range(-5, 6):
                _positions = []
                for y in range(1, -10, -1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Left:

            for x in range(-9, 2):
                _positions = []
                for y in range(-5, 6, 1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Right:

            for x in range(-1, 10):
                _positions = []
                for y in range(-5, 6, 1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        return positions


from components.agent import DirectionType

