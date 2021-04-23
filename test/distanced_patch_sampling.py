import cv2

import env


def main():

    agents = ['blue', 'red']

    tocEnv = env.TOCEnv(agents=agents, map_size=(20, 30))

    while True:

        patch_count = 10
        patch_distance = 6

        tocEnv.set_patch_count(patch_count)
        tocEnv.set_patch_distance(patch_distance)

        tocEnv.reset()
        image = tocEnv.render(coordination=True)

        cv2.imshow('TOCEnv', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
