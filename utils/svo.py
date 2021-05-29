import numpy as np

def svo(rews, aind):
    '''
    rews : all rewards
    aind : agent index
    '''
    exc_rews = (np.sum(rews) - rews[aind]) / len(rews) - 1
    my_rew = rews[aind]
    rew_angle = np.arctan(exc_rews, my_rew)
    target_angle = np.pi/2
    w = 0.2
    U = my_rew - w * abs(target_angle - rew_angle)
    return U

