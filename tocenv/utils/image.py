import numpy as np
import cv2 as cv


def put_rgba_to_image(src, dest, x_offset, y_offset) -> np.array:
    y1, y2 = y_offset, y_offset + src.shape[0]
    x1, x2 = x_offset, x_offset + src.shape[1]

    alpha_s = src[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3): # Loop for BGR channels
        dest[y1:y2, x1:x2, c] = (alpha_s * src[:, :, c] + alpha_l * dest[y1:y2, x1:x2, c])
    return dest


def put_rgb_to_image(src, dest, x_offset, y_offset) -> np.array:
    y1, y2 = y_offset, y_offset + src.shape[0]
    x1, x2 = x_offset, x_offset + src.shape[1]

    for c in range(0, 3): # Loop for BGR channels
        dest[y1:y2, x1:x2, c] = (src[:, :, c] + dest[y1:y2, x1:x2, c])
    return dest
