import abc
from abc import ABC


class Item(object):

    def __init__(self, world) -> None:
        self.world = world

    @abc.abstractmethod
    def on_collapsed(self): pass


class Apple(Item, ABC):
    def __init__(self, world):
        self.world = world
        self.elapsed_step_from_spawned = 0

    def tick(self):
        self.elapsed_step_from_spawned += 1


class RedApple(Apple, ABC):
    def __init__(self, world) -> None:
        super(RedApple, self).__init__(world)


class BlueApple(Apple, ABC):
    def __init__(self, world) -> None:
        super(BlueApple, self).__init__(world)
