import cv2

import env


def main():
    tocEnv = env.TOCEnv(map_size=(20, 30))

    while True:

        patch_count = 3
        patch_distance = 5

        tocEnv.set_patch_count(patch_count)
        tocEnv.set_patch_distance(patch_distance)

        tocEnv.reset()
        image = tocEnv.render()

        cv2.imshow('TOCEnv', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
