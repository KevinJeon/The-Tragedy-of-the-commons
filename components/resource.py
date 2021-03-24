import cv2 as cv
import os


class Resource(object):
    Asset_Dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'assets'))

    Apple = cv.imread(os.path.join(Asset_Dir, 'apple.png'), cv.IMREAD_UNCHANGED)
    Agent = cv.imread(os.path.join(Asset_Dir, 'monster.png'), cv.IMREAD_UNCHANGED)
    Wall = cv.imread(os.path.join(Asset_Dir, 'wall.png'), cv.IMREAD_UNCHANGED)
