#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 23.3.21
"""

import os
import cv2
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR, ROOT_DIR
from multiprocessing.pool import Pool
from myutils.project_utils import *


class DatasetFilterV2(object):
    def __init__(self):
        name = "angle_ds_solution_20210323"
        self.ds_path = os.path.join(DATA_DIR, '{}.txt'.format(name))
        self.filter_path = os.path.join(DATA_DIR, '{}.filter.txt'.format(name))

    def process(self):
        """
        写入文件ß
        :return:
        """
        create_file(self.filter_path)

        data_lines = read_file(self.ds_path)
        out_urls = []
        for data_line in data_lines:
            urls = data_line.split(',')
            url = urls[0]
            out_urls.append(url)

        out_urls = list(set(out_urls))
        print('[Info] 数据行数: {}'.format(len(out_urls)))
        write_list_to_file(self.filter_path, out_urls)  # 写入文件
        print('[Info] 写入完成: {}'.format(self.ds_path))

    @staticmethod
    def filter_img(idx, path, name, out_dir):
        try:
            img_x = cv2.imread(path)
            name_x = name.split('.')[0]
            out_path = os.path.join(out_dir, '{}.jpg'.format(name_x))
            h, w, c = img_x.shape
            if min(h, w) < 8:
                print('[Error] path: {} 最小边: {}'.format(path, min(h, w)))
                return
            if c == 4:
                img_bgr = cv2.cvtColor(img_x, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(out_path, img_bgr)
            elif c == 3:
                img_bgr = img_x
                cv2.imwrite(out_path, img_bgr)
        except Exception as e:
            print('[Error] path: {} {}'.format(path, e))
        print('[Info] {} 处理完成: {}'.format(idx, path))

    def filter(self):
        self.imgs_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'angle_ds_solution_20210323')
        self.out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'angle_ds_solution_20210323_x')
        mkdir_if_not_exist(self.out_dir)
        paths_list, names_list = traverse_dir_files(self.imgs_dir)
        print('[Info] 待处理文件数: {}'.format(len(paths_list)))
        idx = 0
        pool = Pool(processes=100)
        for path, name in zip(paths_list, names_list):
            idx += 1
            # DatasetFilterV2.filter_img(idx, path, name, self.out_dir)
            pool.apply_async(DatasetFilterV2.filter_img, (idx, path, name, self.out_dir))

        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(self.out_dir))


def main():
    df2 = DatasetFilterV2()
    # df2.process()
    df2.filter()


if __name__ == '__main__':
    main()
