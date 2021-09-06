import numpy as np


def svo(rews, aind):
    exc_rews = (np.sum(rews) - rews[aind]) / (len(rews) - 1)
    my_rew = rews[aind]
    if my_rew == 0:
        rew_angle = np.pi / 2
    else:
        rew_angle = np.arctan(exc_rews / my_rew)

    target_angle = np.pi / 2
    w = 0.2
    U = my_rew - w * abs(target_angle - rew_angle)
    return U
