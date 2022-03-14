#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 14.3.22
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from multiprocessing.pool import Pool

from myutils.project_utils import *
from x_utils.oss_utils import save_img_2_oss, get_img_from_oss

from root_dir import DATA_DIR


class Processor20220314(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "files_v3", "all_train_data_0311.txt")
        self.out_file_path = os.path.join(DATA_DIR, "files_v3", "all_train_data_0311.out.txt")
        self.out_folder = "/ProjectRoot/workspace/datasets/rotation_ds_new20220314"
        mkdir_if_not_exist(self.out_folder)

    @staticmethod
    def save_img(img_bgr, img_name):
        oss_object_url = save_img_2_oss(img_bgr, img_name, "zhengsheng.wcl/Image-Rotation/datasets/datasets-20220314/")
        return oss_object_url

    @staticmethod
    def process_line(idx, data_line, out_folder, out_file_path):
        print("[Info] idx: {}".format(idx))
        img_name = str(idx).zfill(6) + ".jpg"
        img_path = os.path.join(out_folder, img_name)
        get_img_from_oss(data_line, img_path)
        write_line(out_file_path, img_path)

    def process(self):
        data_lines = read_file(self.file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))

        pool = Pool(processes=40)
        for idx, data_line in enumerate(data_lines):
            pool.apply_async(Processor20220314.process_line, (idx, data_line, self.out_folder, self.out_file_path))
        pool.close()
        pool.join()


def main():
    pro = Processor20220314()
    pro.process()


if __name__ == '__main__':
    main()
