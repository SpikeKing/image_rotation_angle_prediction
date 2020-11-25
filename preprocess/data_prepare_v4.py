#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 25.11.20
"""

import os
import cv2
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myutils.project_utils import read_file, download_url_img, mkdir_if_not_exist
from myutils.cv_utils import rotate_img_with_bound
from x_utils.vpf_utils import get_problem_rotation_vpf_service


class DataPrepareV4(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, 'sample_complex_formula.txt')
        self.out_dir = os.path.join(DATA_DIR, 'sample_complex_formula_out')
        mkdir_if_not_exist(self.out_dir)

    @staticmethod
    def format_angle(angle):
        """
        格式化角度
        """
        angle = int(angle)
        if angle <= 45 or angle >= 325:
            r_angle = 0
        elif 45 < angle <= 135:
            r_angle = 90
        elif 135 < angle <= 225:
            r_angle = 180
        else:
            r_angle = 270
        return r_angle

    @staticmethod
    def process_url(img_url, out_path):
        is_ok, img_bgr = download_url_img(img_url)
        res_dict = get_problem_rotation_vpf_service(img_url)
        # print('[Info] res: {}'.format(res_dict))
        angle = res_dict['data']['angle']
        # image_oss_url = res_dict['data']['image_oss_url']

        # print('[Info] angle: {}'.format(angle))
        angle = DataPrepareV4.format_angle(angle)
        img_out = rotate_img_with_bound(img_bgr, angle)
        cv2.imwrite(out_path, img_out)
        print('[Info] 存储完成: {}'.format(out_path))

    def process(self):
        data_lines = read_file(self.file_path)
        pool = Pool(processes=40)
        for idx, data_line in enumerate(data_lines):
            # print('[Info] url: {}'.format(data_line))
            out_path = os.path.join(self.out_dir, '{}.jpg'.format(idx))

            pool.apply_async(DataPrepareV4.process_url, (data_line, out_path))
            # DataPrepareV4.process_url(data_line, out_path)
            if idx % 100 == 0:
                print('[Info] idx: {}'.format(idx))

        pool.close()
        pool.join()

        print('[Info] 下载完成')


def main():
    dpv = DataPrepareV4()
    dpv.process()


if __name__ == '__main__':
    main()