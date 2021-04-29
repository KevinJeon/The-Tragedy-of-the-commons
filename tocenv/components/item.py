import abc


class Item(object):

    def __init__(self, world) -> None:
        self.world = world

    @abc.abstractmethod
    def on_collapsed(self): pass


class Apple(Item):

    def __init__(self):
        self.reward = 1.


class RedApple(Apple):
    def __init__(self):
        super(RedApple, self).__init__()
        self.reward = 3.


class BlueApple(Apple):
    def __init__(self):
        super(BlueApple, self).__init__()
        self.reward = 3.
