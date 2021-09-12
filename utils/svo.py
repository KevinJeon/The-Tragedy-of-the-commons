import numpy as np
import math

def svo(rews, aind, preferences):
    if (np.array(rews) == np.zeros(len(rews))).all():
        rew_angle = 45
    else:
        exc_rews = (np.sum(rews) - rews[aind]) / (len(rews) - 1)
        rew_angle = math.atan(exc_rews / int(rews[aind])) * 180 / math.pi
    my_rew = rews[aind]
    preference = preferences[aind]
    w = 0.2
    U = my_rew - w * abs(preference - rew_angle)
    return U
