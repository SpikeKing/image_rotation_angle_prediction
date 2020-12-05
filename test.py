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
    img_path = os.path.join(DATA_DIR, 'error_imgs', 'error1_20201125.0.jpg')
    im = cv2.imread(img_path)
    show_img_bgr(im)
    angle = 90
    new_im1 = rotate_img_for_4angle(im, (360 - angle) % 360)
    new_im2 = rotate_img_with_bound(im, angle)  # 第2种旋转模型
    new_im3, angle, rhw_ratio = generate_rotated_image(im, angle)

    show_img_bgr(new_im1)
    show_img_bgr(new_im2)
    show_img_bgr(new_im3)


if __name__ == '__main__':
    main()