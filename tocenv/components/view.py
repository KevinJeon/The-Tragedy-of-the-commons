
from tocenv.components.position import Position
from tocenv.components.agent import DirectionType
import math

class View(object):
    def __init__(self):
        pass

    def get_visible_positions(direction: DirectionType) -> [Position]:
        pass


from tocenv.components.position import Position
from tocenv.components.agent import DirectionType


class View(object):

    @staticmethod
    def get_visible_positions(direction: DirectionType) -> [Position]:
        direction = direction.get_type()
        obs_dim = 11
        positions = []
        if direction == DirectionType.Up:

            for y in range(obs_dim - 2, -2, -1):
                _positions = []
                for x in range(-math.trunc(obs_dim/2), math.ceil(obs_dim/2)):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Down:
            for y in range(-obs_dim + 2, 2, 1):
                _positions = []
                for x in range(math.trunc(obs_dim/2), -math.ceil(obs_dim/2), -1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Left:

            for x in range(-obs_dim + 2, 2):
                _positions = []
                for y in range(-math.trunc(obs_dim/2), math.ceil(obs_dim/2), 1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        elif direction == DirectionType.Right:

            for x in range(obs_dim - 2, -2, -1):
                _positions = []
                for y in range(math.trunc(obs_dim/2), -math.ceil(obs_dim/2), -1):
                    _positions.append(Position(x=x, y=y))
                positions.append(_positions)

        return positions

    @staticmethod
    def get_agent_position() -> Position:
        return Position(x=5, y=1)



