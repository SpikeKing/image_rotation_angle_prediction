#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.5.21
"""

import os
import cv2

from root_dir import DATA_DIR
from myutils.cv_utils import show_img_bgr


def center_crop_by_hw(img_bgr):
    """
    避免图像的比例失衡
    """
    h, w, _ = img_bgr.shape
    if h // w > 3:
        mid = h // 2
        img_crop = img_bgr[mid-w:mid+w, :, :]
        return img_crop
    if w // h > 3:
        mid = w // 2
        img_crop = img_bgr[:, mid-h:mid+h, :]
        return img_crop
    else:
        return img_bgr

def main():
    img_path = os.path.join(DATA_DIR, 'cases', '1.90.jpg')
    img_bgr = cv2.imread(img_path)
    img_bgr = center_crop_by_hw(img_bgr)
    show_img_bgr(img_bgr)


if __name__ == '__main__':
    main()
