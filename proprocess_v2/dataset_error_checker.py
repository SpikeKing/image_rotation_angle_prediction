#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 6.9.21
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR, DATA_DIR
from myutils.project_utils import *


class DatasetErrorChecker(object):
    def __init__(self):
        self.out_img_paths_file = os.path.join(DATA_DIR, "files_v2", "dataset_all_path_{}.txt".format(get_current_time_str()))

    @staticmethod
    def process_line(folder_path, out_file):
        print('[Info] 读取路径: {}'.format(folder_path))
        paths_list, names_list = traverse_dir_files(folder_path)
        print('[Info] 读取完成: {}'.format(len(paths_list)))
        write_list_to_file(out_file, paths_list)
        print('[Info] 写入完成: {}, 样本数: {}'.format(out_file, len(paths_list)))

    def load_dataset(self):
        dataset13_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_nat_v2_raw_20210829')
        dataset12_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_nat_20210828')
        dataset11_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_table_20210828')
        dataset10_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_trans_20210828')
        dataset9_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_other')
        dataset8_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_xiaotu_25w')
        dataset1_path = os.path.join(ROOT_DIR, '..', 'datasets', 'segmentation_ds_v4', 'images')
        dataset2_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked_r')
        dataset3_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_tiku_5k')
        dataset4_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_page_2w')
        dataset5_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_write_4w')
        dataset6_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_write2_3w')
        dataset7_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_page_bkg_2w')
        dataset_val_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_val')

        folder_path_list = [dataset1_path, dataset2_path, dataset3_path, dataset4_path, dataset5_path,
                            dataset6_path, dataset7_path, dataset8_path, dataset9_path, dataset10_path,
                            dataset11_path, dataset12_path, dataset13_path, dataset_val_path]

        for folder_path in folder_path_list:
            DatasetErrorChecker.process_line(folder_path, self.out_img_paths_file)
        print('[Info] 写入完成: {}'.format(self.out_img_paths_file))

    def check(self):
        pass


def main():
    dc = DatasetErrorChecker()
    dc.load_dataset()


if __name__ == '__main__':
    main()
