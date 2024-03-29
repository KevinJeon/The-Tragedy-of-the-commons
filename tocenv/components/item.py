import abc
from abc import ABC


class Item(object):

    def __init__(self, world) -> None:
        self.world = world

    @abc.abstractmethod
    def on_collapsed(self): pass


class Apple(Item, ABC):
    def __init__(self):
        self.reward = 1


class RedApple(Apple, ABC):
    def __init__(self):
        super(RedApple, self).__init__()


class BlueApple(Apple, ABC):
    def __init__(self):
        super(BlueApple, self).__init__()
