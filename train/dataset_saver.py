#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 14.9.21
"""

import os
import sys
from multiprocessing import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class DatasetSaver(object):
    """
    预存数据文件，避免遍历速度过慢
    """
    def __init__(self):
        self.data_file = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_{}.txt".format(get_current_day_str()))
        self.data_folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_{}".format(get_current_day_str()))

    @staticmethod
    def process_line(folder_path, num, out_file=None, out_folder=None):
        print('[Info] 读取路径: {}'.format(folder_path))
        paths_list, names_list = traverse_dir_files(folder_path, is_sorted=False, ext="jpg")
        print('[Info] 读取完成: {}'.format(len(paths_list)))
        paths_list = format_samples_num(paths_list, num)
        if out_file:
            write_list_to_file(out_file, paths_list)
        else:
            mkdir_if_not_exist(out_folder)
            folder_name = folder_path.split("/")[-1]
            file_path = os.path.join(out_folder, "{}_{}.txt".format(folder_name, get_current_day_str()))
            create_file(file_path)  # 避免重复写入
            write_list_to_file(file_path, paths_list)
        print('[Info] 写入完成: {}, 样本数: {}'.format(folder_path, len(paths_list)))

    def load_dataset_mul_file(self):
        s_time = time.time()
        # hardcase, 112
        dataset14_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_hardcase')
        # 自然场景图像, 71885
        dataset13_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_nat_v2_raw_20210829_1024')
        # 自然场景图像, 13961
        dataset12_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_nat_20210828_1024')
        # 表格图像, 42579
        dataset11_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_table_20210828')
        # 图像翻译图像, 88986
        dataset10_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_trans_20210828')
        # 其他图像, 2168
        dataset9_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_other_1024')
        # 24w小图数据集, 238927
        dataset8_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_xiaotu_25w_512')
        # 14w query数据, 已验证, 147756
        dataset1_path = os.path.join(ROOT_DIR, '..', 'datasets', 'segmentation_ds_v4')
        # 12w query数据, 130989
        dataset2_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked_r')
        # 5k 题库数据, 4963
        dataset3_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_tiku_5k')
        # 2w 整页拍数据, 21248
        dataset4_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_page_2w')
        # 4w 手写数据, 37548
        dataset5_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_write_4w')
        # 3w 手写数据, 28429
        dataset6_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_write2_3w')
        # 2.2w 题库修改数据, 21169
        dataset7_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_page_bkg_2w')

        dataset_num_list = \
            [[dataset14_path, -1],
             [dataset13_path, -1], [dataset12_path, -1], [dataset11_path, -1], [dataset10_path, -1], [dataset9_path, -1],
             [dataset8_path, -1], [dataset1_path, -1], [dataset2_path, -1], [dataset3_path, -1],
             [dataset4_path, -1], [dataset5_path, -1], [dataset6_path, -1], [dataset7_path, -1]]

        pool = Pool(processes=100)
        for folder_path, num in dataset_num_list:
            pool.apply_async(DatasetSaver.process_line, (folder_path, num, None, self.data_folder))  # 使用folder
        pool.close()
        pool.join()
        print('[Info] 写入完成: {}'.format(self.data_file))
        elapsed_time = time_elapsed(s_time, time.time())
        print('[Info] 耗时: {}'.format(elapsed_time))

    def load_dataset_mul_file_v2(self):
        s_time = time.time()
        dataset1_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_nat_roi')
        dataset2_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_nat_tl')
        dataset_num_list = [[dataset1_path, -1], [dataset2_path, -1]]
        pool = Pool(processes=100)
        for folder_path, num in dataset_num_list:
            pool.apply_async(DatasetSaver.process_line, (folder_path, num, None, self.data_folder))  # 使用folder
        pool.close()
        pool.join()
        print('[Info] 写入完成: {}'.format(self.data_file))
        elapsed_time = time_elapsed(s_time, time.time())
        print('[Info] 耗时: {}'.format(elapsed_time))

    @staticmethod
    def load_dataset_one_file():
        dataset1_path = os.path.join(ROOT_DIR, '..', 'datasets', 'text_line_v1_200w')
        print('[Info] 处理文件夹: {}'.format(dataset1_path))
        out_file = os.path.join(DATA_DIR, "files_v2", "text_line_v1_200w_path.txt")
        print('[Info] 输出文件: {}'.format(out_file))
        s_time = time.time()
        paths_list, names_list = traverse_dir_files(dataset1_path, is_sorted=False)
        print('[Info] 文件行数: {}'.format(len(paths_list)))
        print("[Info] time: {}".format(time_elapsed(s_time, time.time())))
        write_list_to_file(out_file, paths_list)
        print('[Info] 写入文件: {}'.format(out_file))


def main():
    ds = DatasetSaver()
    ds.load_dataset_mul_file_v2()


if __name__ == '__main__':
    main()
