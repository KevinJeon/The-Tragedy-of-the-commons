import abc


class Item(object):

    def __init__(self, world) -> None:
        self.world = world

    @abc.abstractmethod
    def on_collapsed(self): pass


class Apple(Item):

    def __init__(self):
        self.reward = 1.









