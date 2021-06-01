import numpy as np


def ma_obs_to_numpy(arr_obs) -> np.array:
    return np.array(arr_obs).transpose(0, 3, 1, 2)