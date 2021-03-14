import cv2 as cv
import random


from env import TOCEnv
from components.agent import Action

def main():
    env = TOCEnv()

    while True:
        _ = env.reset()

        for i in range(100):
            actions = [random.randint(0, 3) for _ in range(4)]
            env.step(actions=actions)
            image = env.render()
            cv.imshow('Env', image)
            cv.waitKey(0)


if __name__ == '__main__':
    main()