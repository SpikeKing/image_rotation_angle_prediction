#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 26.11.20
"""

import os
import cv2
import numpy as np

from root_dir import DATA_DIR
from myutils.cv_utils import *
from utils import generate_rotated_image


def main():
    img_path = os.path.join(DATA_DIR, 'cases', '100280_v.jpg')
    out1_path = os.path.join(DATA_DIR, 'cases', '100280_v.out1.jpg')
    out2_path = os.path.join(DATA_DIR, 'cases', '100280_v.out2.jpg')

    im = cv2.imread(img_path)
    # cv2.imwrite(out_path, im)
    # show_img_bgr(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    im = im.astype(np.uint8)
    # show_img_bgr(im)
    angle = 90
    new_im1 = rotate_img_for_4angle(im, (360 - angle) % 360)
    cv2.imwrite(out1_path, new_im1)

    # new_im2 = rotate_img_with_bound(im, angle)  # 第2种旋转模型

    new_im2, angle, rhw_ratio = generate_rotated_image(new_im1, 10, crop_largest_rect=True)
    cv2.imwrite(out2_path, new_im2)

    # show_img_bgr(new_im1)
    # show_img_bgr(new_im2)
    # show_img_bgr(new_im3)


def main_v2():
    img_path = os.path.join(DATA_DIR, 'cases', '100280_v.jpg')
    out3_path = os.path.join(DATA_DIR, 'cases', '100280_v.out4.jpg')
    image = cv2.imread(img_path)

    # h, w, _ = image.shape
    # out_h = int(h // 2)  # mode 1
    # image = random_crop(image, out_h, w)

    h, w, _ = image.shape
    out_w = int(w // 2)  # mode 1
    image = random_crop(image, h, out_w)

    cv2.imwrite(out3_path, image)


if __name__ == '__main__':
    # main()
    main_v2()
