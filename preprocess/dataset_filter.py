#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 30.11.20
"""

import os
import sys
import cv2
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class DatasetFilter(object):
    def __init__(self):
        pass

    def filter(self):
        data_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf.right.txt')
        out_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf_right')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(data_file)
        for data_line in data_lines:
            url, angle = data_line.split(',')
            out_path = os.path.join(out_dir, 'angle_{}.txt'.format(angle))
            write_line(out_path, data_line)

    @staticmethod
    def process_img_angle(idx, url, angle, out_dir):
        try:
            angle = int(angle)
            name = url.split('/')[-1]
            is_ok, img_bgr = download_url_img(url)
            img_out = rotate_img_for_4angle(img_bgr, angle)
            out_path = os.path.join(out_dir, "{}".format(name))
            cv2.imwrite(out_path, img_out)
        except Exception as e:
            print('[Error] {} 错误'.format(idx))
            return
        print('[Info] {} {} 完成'.format(idx, out_path))

    def download_right_angle(self):
        files_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf_right')
        paths_list, names_list = traverse_dir_files(files_dir)

        pool = Pool(processes=80)
        for path, name in zip(paths_list, names_list):
            name_x = name.split('.')[0]
            urls_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf_right', '{}.txt'.format(name_x))  # 输入
            out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked', 'vpf_right', name_x)  # 输出
            mkdir_if_not_exist(out_dir)

            data_lines = read_file(urls_file)

            for idx, data_line in enumerate(data_lines):
                url, angle = data_line.split(',')
                pool.apply_async(DatasetFilter.process_img_angle, (idx, url, angle, out_dir))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(files_dir))


def main():
    df = DatasetFilter()
    # df.filter()
    df.download_right_angle()


if __name__ == '__main__':
    main()