#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 25.11.20
"""

import os
import cv2

from myutils.project_utils import *
from root_dir import DATA_DIR


class RealEvaluation(object):
    def __init__(self):
        pass

    def process(self):
        file_path = os.path.join(DATA_DIR, 'test_1000.txt')
        out_dir = os.path.join(DATA_DIR, 'test_1000_out')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(file_path)
        for idx, data_line in enumerate(data_lines):
            img_url, angle = data_line.split('\t')
            print('[Info] img_url: {}, angle: {}'.format(img_url, angle))
            is_ok, img_bgr = download_url_img(img_url)
            if angle == 90:
                img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
            elif angle == 270:
                img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img_rotated = img_bgr

            out_path = os.path.join(out_dir, '{}.jpg'.format(idx))
            cv2.imwrite(out_path, img_rotated)
            print('[Info] 完成: {}'.format(out_path))


def main():
    reo = RealEvaluation()
    reo.process()


if __name__ == '__main__':
    main()