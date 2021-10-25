#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 25.10.21

重新构建数集
"""

import os
import shutil
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR, DATA_DIR
from x_utils.vpf_sevices import get_vpf_service_np
from myutils.project_utils import *
from myutils.cv_utils import *


class DatasetReorder(object):
    """
    重新构建数据集
    """
    def __init__(self):
        self.folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_20211021")
        self.out_files_folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_20211025")
        self.out_ds_folder = os.path.join(ROOT_DIR, "..", "datasets", "angle_datasets")

    def merge_hardcase(self, file_list, type_name):
        # file1 = os.path.join(self.folder, "rotation_datasets_hardcase.112.txt")
        # file2 = os.path.join(self.folder, "rotation_ds_other_1024_20210927.txt")
        # type_name = "hardcase"
        data_lines = []
        for file_name in file_list:
            file_path = os.path.join(self.folder, file_name)
            data_lines = read_file(file_path)
            data_lines += data_lines
        print('[Info] 样本行数: {}'.format(len(data_lines)))

        folder_name = "dataset_{}_{}".format(type_name, len(data_lines))
        dataset_folder = os.path.join(self.out_ds_folder, folder_name)
        print('[Info] 输出文件夹路径: {}'.format(dataset_folder))
        out_path_file = os.path.join(self.out_files_folder, "{}.txt".format(folder_name))
        print('[Info] 输出文件路径: {}'.format(out_path_file))

        mkdir_if_not_exist(dataset_folder)
        path_list = []
        for data_idx, data_line in enumerate(data_lines):
            data_idx_str = str(data_idx).zfill(6)
            out_name = "{}_{}.jpg".format(type_name, data_idx_str)
            out_path = os.path.join(dataset_folder, out_name)
            shutil.copy(data_line, out_path)
            path_list.append(out_path)
            if data_idx % 5000 == 0:
                print('[Info] \t{}'.format(data_idx))
        write_list_to_file(out_path_file, path_list)
        print('[Info] 输出路径: {}, 样本数: {}'.format(len(path_list), len(data_lines)))
        print('[Info] 处理完成: {}'.format(out_path_file))

    def process(self):
        file_list = ["rotation_datasets_hardcase.112.txt", "rotation_ds_other_1024_20210927.txt"]
        type_name = "hardcase"
        self.merge_hardcase(file_list, type_name)


def main():
    dr = DatasetReorder()
    dr.process()


if __name__ == '__main__':
    main()
