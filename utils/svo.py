import numpy as np
import math

def svo(rews, aind, preferences):
    exc_rews = (np.sum(rews) - rews[aind]) / (len(rews) - 1)
    my_rew = rews[aind]
    preference = preferences[aind]
    rew_angle = math.atan(exc_rews / rews[aind]) * 180 / math.pi
    w = 0.2
    U = my_rew - w * abs(preference - rew_angle)
    return U
