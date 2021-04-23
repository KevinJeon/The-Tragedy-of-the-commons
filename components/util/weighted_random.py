import numpy as np
import random

from components.position import Position


def get_weighted_position(mu: float, sigma: float, map_size: tuple) -> Position:
    '''
    Get a weighted random position on the world

    :param mu: Normal distribution's mu
    :param sigma: Normal distribution's sigma
    :param map_size: (height, width)
    :return: Weighted random position on the world
    '''

    sampled = np.random.normal(mu, sigma, size=1000)
    scale = max(abs(min(sampled)), abs(max(sampled)))

    x = random.choice(np.array((sampled / scale * 0.5 + 0.5) * map_size[1], dtype=np.int))
    y = random.choice(np.array((sampled / scale * 0.5 + 0.5) * map_size[0], dtype=np.int))

    return Position(x=x, y=y)
