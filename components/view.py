from components.world import Position
from components.agent import DirectionType


class View(object):

    @staticmethod
    def get_visible_positions(direction: DirectionType) -> [Position]:
        direction = direction.get_type()

        positions = []
        if direction == DirectionType.Up:

            for x in range(-5, 6):
                for y in range(9, -2, -1):
                    positions.append(Position(x=x, y=y))

        elif direction == DirectionType.Down:

            for x in range(-5, 6):
                for y in range(1, -10, -1):
                    positions.append(Position(x=x, y=y))

        elif direction == DirectionType.Left:

            for x in range(-9, 2):
                for y in range(-5, 6, 1):
                    positions.append(Position(x=x, y=y))
        elif direction == DirectionType.Right:

            for x in range(-1, 10):
                for y in range(-5, 6, 1):
                    positions.append(Position(x=x, y=y))

        return positions
