


from tocenv.components.position import Position


class DirectionType:
    # Clock-wise numbering
    Up = 0
    Down = 2
    Left = 1
    Right = 3



class Direction(object):
    def __init__(self):
        pass


class Direction(object):

    def __init__(self, direction_type):
        self.direction = direction_type

    def turn_right(self) -> Direction:
        self.direction = (self.direction + 1) % 4
        return self

    def turn_left(self) -> Direction:
        self.direction = ((self.direction + 4) - 1) % 4
        return self

    def half_rotate(self) -> Direction:
        self.direction = (self.direction + 2) % 4
        return self

    @property
    def value(self):
        return self.direction

    def _to_position(self) -> Position:
        if self.direction == DirectionType.Up:
            return Position(x=0, y=1)
        elif self.direction == DirectionType.Down:
            return Position(x=0, y=-1)
        elif self.direction == DirectionType.Left:
            return Position(x=-1, y=0)
        elif self.direction == DirectionType.Right:
            return Position(x=1, y=0)

    def _to_string(self) -> str:
        if self.direction == DirectionType.Up:
            return 'Up'
        elif self.direction == DirectionType.Down:
            return 'Down'
        elif self.direction == DirectionType.Left:
            return 'Left'
        elif self.direction == DirectionType.Right:
            return 'Right'

    def get_type(self):
        return self.direction

    def __str__(self):
        return 'Direction({0})'.format(self._to_string())

