import math


class DistanceType:
    Euclidean = 1
    Manhattan = 2


class Position(object):
    def __init__(self, x=None, y=None):
        pass


class Position(object):
    def __init__(self, x=None, y=None):
        assert x is not None
        assert y is not None
        self.x = x
        self.y = y

    def subtract_y(self, scalar: int) -> Position:
        return Position(x=self.x, y=scalar - self.y)

    def __add__(self, pos: Position):
        return Position(x=self.x + pos.x, y=self.y + pos.y)

    def __sub__(self, pos: Position):
        return Position(x=self.x - pos.x, y=self.y - pos.y)

    def __mul__(self, scale: int):
        return Position(x=self.x * scale, y=self.y * scale)

    def __eq__(self, other: Position):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return 'Position(x={0}, y={1})'.format(self.x, self.y)

    def __hash__(self):
        return self.x * 99999 + self.y  # Cheat number

    def to_tuple(self, reverse=False):
        if reverse:
            return self.x, self.y
        else:
            return self.y, self.x

    def get_distance(self, pos: Position, distance_type=DistanceType):
        if distance_type == DistanceType.Euclidean:
            return math.sqrt(math.pow(self.x - pos.x, 2) + math.pow(self.y - pos.y, 2))
        else:  # Manhattan distance
            return abs(self.x - pos.x) + abs(self.y - pos.y)

    def get_surrounded(self, radius: int) -> [Position]:
        surrounded = []
        for _y in range(radius * -1, radius + 1):
            for _x in range(radius * -1, radius + 1):
                _position = self + Position(x=_x, y=_y)
                distance = self.get_distance(_position)

                if distance <= radius:
                    surrounded.append(_position)

        return surrounded
