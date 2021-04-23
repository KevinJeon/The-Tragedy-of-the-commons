import cv2

import env


def main():

    agents = ['blue', 'red']

    tocEnv = env.TOCEnv(agents=agents, map_size=(20, 20))

    patch_count = 10
    patch_distance = 7

    for count in range(5, patch_count):
        for distance in range(4, patch_distance):
            for _ in range(100):
                tocEnv.set_patch_count(count)
                tocEnv.set_patch_distance(distance)
                tocEnv.set_apple_color_ratio(0.5)
                tocEnv.apple_spawn_ratio(0)
                ret = tocEnv.reset()
                # print(ret)
                image = tocEnv.render(coordination=True)
                cv2.imshow('TOCEnv', image)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()
