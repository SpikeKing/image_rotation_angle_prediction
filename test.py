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


def main():
    img_path = os.path.join(DATA_DIR, 'error_imgs', 'error1_20201125.0.jpg')
    im = cv2.imread(img_path)
    new_im = rotate_img_for_4angle(im, (360 - 90) % 360)
    # new_im = rotate_img_with_bound(im, 90)
    show_img_bgr(new_im)


if __name__ == '__main__':
    main()