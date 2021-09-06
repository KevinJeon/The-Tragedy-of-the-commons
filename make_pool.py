import numpy as np


def make_pool(avg_prefer, num_agent):
    '''
    we think avg_prefer as average of blue_prefer
    '''
    max_prefer = 4
    rands = np.random.uniform(0, max_prefer - avg_prefer, num_agent)
    rands = rands - np.sum(rands)/num_agent + avg_prefer

    return rands
    

