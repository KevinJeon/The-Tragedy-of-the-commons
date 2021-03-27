from components.position import Position
from components.agent import DirectionType


class View(object):

    @staticmethod
    def get_visible_positions(direction: DirectionType) -> [Position]:
        direction = direction.get_type()

        positions = []
        if direction == DirectionType.Up:

            for y in range(9, -2, -1):
                _positions = []
                for x in range(-5, 6):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Down:
            for y in range(-9, 2, 1):
                _positions = []
                for x in range(-5, 6):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Left:

            for x in range(-9, 2):
                _positions = []
                for y in range(-5, 6, 1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Right:

            for x in range(9, -2, -1):
                _positions = []
                for y in range(5, -6, -1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        return positions

    @staticmethod
    def get_agent_position() -> Position:
        return Position(x=5, y=1)


from components.agent import DirectionType

