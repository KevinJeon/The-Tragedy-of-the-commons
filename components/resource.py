import numpy as np
import cv2 as cv
import os

Asset_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'assets'))

def load_asset(filename):
    global Asset_Dir

    f = np.fromfile(os.path.join(Asset_Dir, filename), np.uint8)
    img = cv.imdecode(f, cv.IMREAD_UNCHANGED)
    return img


class Resource(object):

    Apple = load_asset('apple.png')
    AppleRed = load_asset('apple.png')
    AppleBlue = load_asset('apple_blue.png')
    Agent = load_asset('monster.png')
    AgentRed = load_asset('monster2.png')
    AgentBlue = load_asset('monster2.png')
    Wall = load_asset('wall.png')
    Flame = load_asset('flame.png')
