import cv2 as cv

from env import TOCEnv

def main():
    env = TOCEnv()

    while True:
        state = env.reset()



if __name__ == '__main__':
    main()