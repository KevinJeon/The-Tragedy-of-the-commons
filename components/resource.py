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
    Apple = cv.imread(os.path.join(Asset_Dir, 'apple.png'), cv.IMREAD_UNCHANGED)
    AppleRed = cv.imread(os.path.join(Asset_Dir, 'apple_red.png'), cv.IMREAD_UNCHANGED)
    AppleBlue = cv.imread(os.path.join(Asset_Dir, 'apple_blue.png'), cv.IMREAD_UNCHANGED)
    Agent = cv.imread(os.path.join(Asset_Dir, 'monster.png'), cv.IMREAD_UNCHANGED)
    AgentRed = cv.imread(os.path.join(Asset_Dir, 'monster_red.png'), cv.IMREAD_UNCHANGED)
    AgentBlue = cv.imread(os.path.join(Asset_Dir, 'monster_blue.png'), cv.IMREAD_UNCHANGED)
    Wall = cv.imread(os.path.join(Asset_Dir, 'wall.png'), cv.IMREAD_UNCHANGED)
    Flame = cv.imread(os.path.join(Asset_Dir, 'flame.png'), cv.IMREAD_UNCHANGED)
