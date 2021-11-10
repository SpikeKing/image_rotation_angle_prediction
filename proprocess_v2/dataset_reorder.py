#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 25.10.21

重新构建数集
"""

import os
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR, DATA_DIR
from myutils.project_utils import *
from myutils.cv_utils import *


class DatasetReorder(object):
    """
    重新构建数据集
    """
    def __init__(self):
        self.folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_20211021")
        self.out_files_folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_20211026_raw")
        self.out2_files_folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_20211026")
        self.out3_files_folder = os.path.join(DATA_DIR, "files_v2", "angle_dataset_val_20211026")
        self.out_ds_folder = os.path.join(ROOT_DIR, "..", "datasets", "angle_datasets")

    @staticmethod
    def copy_line_mul(data_idx, data_line, type_name, dataset_folder, out_path_file):
        data_idx_str = str(data_idx).zfill(6)
        out_name = "{}_{}.jpg".format(type_name, data_idx_str)
        out_path = os.path.join(dataset_folder, out_name)
        shutil.copy(data_line, out_path)
        write_line(out_path_file, out_path)
        if data_idx % 2000 == 0:
            print('[Info] \t{}'.format(data_idx))

    def merge_hardcase(self, file_list, type_name):
        data_lines = []
        for file_name in file_list:
            file_path = os.path.join(self.folder, file_name)
            sub_lines = read_file(file_path)
            data_lines += sub_lines
        print('[Info] 样本行数: {}'.format(len(data_lines)))

        folder_name = "dataset_{}_{}".format(type_name, len(data_lines))
        dataset_folder = os.path.join(self.out_ds_folder, folder_name)
        mkdir_if_not_exist(dataset_folder)
        print('[Info] 输出文件夹路径: {}'.format(dataset_folder))

        mkdir_if_not_exist(self.out_files_folder)
        out_path_file = os.path.join(self.out_files_folder, "{}.txt".format(folder_name))
        print('[Info] 输出文件路径: {}'.format(out_path_file))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(
                DatasetReorder.copy_line_mul, (data_idx, data_line, type_name, dataset_folder, out_path_file))
        pool.close()
        pool.join()
        path_list = read_file(out_path_file)
        print('[Info] 输出路径: {}, 样本数: {}'.format(len(path_list), len(data_lines)))
        print('[Info] 处理完成: {}'.format(out_path_file))

    def process(self):
        file_list = ["rotation_datasets_hardcase.112.txt", "rotation_ds_other_1024_20210927.txt"]
        type_name = "hardcase"
        self.merge_hardcase(file_list, type_name)

        file_list = ["datasets_v4_checked_r.131k.txt", "segmentation_ds_v4_20210927.txt"]
        type_name = "query"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_datasets_table.45k.txt"]
        type_name = "table"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_datasets_trans.93k.txt"]
        type_name = "translation"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_ds_write2_3w_20210927.txt", "rotation_ds_write_4w_20210927.txt"]
        type_name = "handwrite"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_ds_xiaotu_25w_512_20210927.txt"]
        type_name = "little-symbol"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_ds_tiku_5k_20210927.txt", "rotation_ds_page_2w_20210927.txt", "rotation_ds_page_bkg_2w_20210927.txt"]
        type_name = "fullpage"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_datasets_nat_v2_raw.75k.txt", "rotation_datasets_nat.14k.txt"]
        type_name = "nature"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_ds_nat_roi_20211021.txt"]
        type_name = "nature-roi"
        self.merge_hardcase(file_list, type_name)

        file_list = ["rotation_ds_nat_tl_20211021.txt"]
        type_name = "nature-textline"
        self.merge_hardcase(file_list, type_name)

    def process_v2(self):
        val_folder = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_val')
        data_lines, _ = traverse_dir_files(val_folder)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        type_name = "val"

        folder_name = "dataset_{}_{}".format(type_name, len(data_lines))
        dataset_folder = os.path.join(self.out_ds_folder, folder_name)
        mkdir_if_not_exist(dataset_folder)
        print('[Info] 输出文件夹路径: {}'.format(dataset_folder))

        mkdir_if_not_exist(self.out_files_folder)
        out_path_file = os.path.join(self.out_files_folder, "{}.txt".format(folder_name))
        print('[Info] 输出文件路径: {}'.format(out_path_file))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(
                DatasetReorder.copy_line_mul, (data_idx, data_line, type_name, dataset_folder, out_path_file))

        pool.close()
        pool.join()
        path_list = read_file(out_path_file)
        print('[Info] 输出路径: {}, 样本数: {}'.format(len(path_list), len(data_lines)))
        print('[Info] 处理完成: {}'.format(out_path_file))

    def format_samples(self, in_file, sample_num):
        in_path = os.path.join(self.out_files_folder, in_file)
        print('[Info] 路径: {}'.format(in_path))
        data_lines = read_file(in_path)
        data_lines = get_fixed_samples(data_lines, sample_num)
        n_data = len(data_lines)
        print('[Info] 样本数: {}'.format(n_data))
        out_file = "_".join(in_file.split("_")[:-1] + [str(n_data)]) + ".txt"
        mkdir_if_not_exist(self.out2_files_folder)
        out_path = os.path.join(self.out2_files_folder, out_file)
        print('[Info] 输出路径: {}'.format(out_path))
        create_file(out_path)
        write_list_to_file(out_path, data_lines)
        print('[Info] 写入完成: {}'.format(out_file))

    def format_samples_val(self, in_file, sample_num):
        in_path = os.path.join(self.out2_files_folder, in_file)
        print('[Info] 路径: {}'.format(in_file))
        data_lines = read_file(in_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        data_lines = get_fixed_samples(data_lines, sample_num)
        n_data = len(data_lines)
        print('[Info] 样本数: {}'.format(n_data))
        out_file = "_".join(in_file.split("_")[:-1] + [str(n_data)]) + ".val.txt"
        mkdir_if_not_exist(self.out3_files_folder)
        out_path = os.path.join(self.out3_files_folder, out_file)
        print('[Info] 输出路径: {}'.format(out_path))
        create_file(out_path)
        write_list_to_file(out_path, data_lines)
        print('[Info] 写入完成: {}'.format(out_file))

    def process_v3(self):
        self.format_samples("dataset_fullpage_49614.txt", 50000)
        self.format_samples("dataset_handwrite_69449.txt", 70000)
        self.format_samples("dataset_hardcase_2280.txt", 3000)
        self.format_samples("dataset_little-symbol_238753.txt", 50000)
        self.format_samples("dataset_nature_89507.txt", 90000)
        self.format_samples("dataset_nature-roi_113430.txt", 120000)
        self.format_samples("dataset_nature-textline_44338.txt", 50000)
        self.format_samples("dataset_query_278744.txt", 100000)
        self.format_samples("dataset_table_44812.txt", 50000)
        self.format_samples("dataset_translation_93670.txt", 50000)
        self.format_samples("dataset_val_1054.txt", 2000)

    def process_v4(self):
        self.format_samples_val("dataset_fullpage_50000.txt", 3000)
        self.format_samples_val("dataset_handwrite_70000.txt", 3000)
        self.format_samples_val("dataset_hardcase_3000.txt", 2000)
        self.format_samples_val("dataset_little-symbol_50000.txt", 3000)
        self.format_samples_val("dataset_nature_90000.txt", 3000)
        self.format_samples_val("dataset_nature-roi_120000.txt", 3000)
        self.format_samples_val("dataset_nature-textline_50000.txt", 3000)
        self.format_samples_val("dataset_query_100000.txt", 3000)
        self.format_samples_val("dataset_table_50000.txt", 3000)
        self.format_samples_val("dataset_translation_50000.txt", 3000)
        self.format_samples_val("dataset_val_2000.txt", 1000)

    def process_v5(self):
        folder = os.path.join(DATA_DIR, "files_v2", "text_line_folder")
        urls_format = os.path.join(DATA_DIR, "files_v2", "urls_textline_{}.txt")
        paths_list, _ = traverse_dir_files(folder)
        data_lines = []
        for path in paths_list:
            sub_lines = read_file(path)
            data_lines += sub_lines
        print('[Info] 样本数: {}'.format(len(data_lines)))
        num = (len(data_lines) // 10000 + 1) * 10000
        data_lines = get_fixed_samples(data_lines, num)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        file_path = urls_format.format(len(data_lines))
        write_list_to_file(file_path, data_lines)
        print('[Info] urls 写入完成: {}'.format(file_path))

    @staticmethod
    def download_line_mul(data_idx, data_line, type_name, dataset_folder, out_path_file):
        data_idx_str = str(data_idx).zfill(6)
        out_name = "{}_{}.jpg".format(type_name, data_idx_str)
        _, img_bgr = download_url_img(data_line)
        img_path = os.path.join(dataset_folder, out_name)
        write_line(out_path_file, img_path)
        cv2.imwrite(img_path, img_bgr)
        if data_idx % 2000 == 0:
            print('[Info] \t{}'.format(data_idx))

    def process_v6(self):
        file_path = os.path.join(DATA_DIR, "files_v2", "urls_handwrite-v2_raw.txt")
        print('[Info] 处理文件: {}'.format(file_path))
        type_name = "handwrite-v2-raw"
        data_lines = read_file(file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))

        # 输出文件夹和输出路径
        dataset_folder = os.path.join(self.out_ds_folder, "dataset_handwrite-v2-raw_{}".format(len(data_lines)))
        mkdir_if_not_exist(dataset_folder)
        out_path_file = os.path.join(self.out_files_folder, "dataset_handwrite-v2-raw_{}.txt".format(len(data_lines)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(DatasetReorder.download_line_mul,
                             (data_idx, data_line, type_name, dataset_folder, out_path_file))

        pool.close()
        pool.join()
        path_list = read_file(out_path_file)
        print('[Info] 输出路径: {}, 样本数: {}'.format(len(path_list), len(data_lines)))
        print('[Info] 处理完成: {}'.format(out_path_file))

    def merge_val_cases(self):
        print('[Info] 验证文件夹: {}'.format(self.out3_files_folder))
        paths_list, names_list = traverse_dir_files(self.out3_files_folder)
        out_file_format = os.path.join(self.out2_files_folder, "dataset_merged_val_{}.txt")
        out_lines = []
        for path in paths_list:
            data_lines = read_file(path)
            out_lines += data_lines
        print('[Info] 验证样本数: {}'.format(out_lines))
        out_lines = out_lines * 4
        print('[Info] 验证样本数: {}'.format(out_lines))
        out_file = out_file_format.format(len(out_lines))
        write_list_to_file(out_file, out_lines)
        print('[Info] 写入完成: {}'.format(out_file))

    def process_v7(self):
        self.format_samples("dataset_textline_195043.txt", 200000)
        self.format_samples_val("dataset_textline_200000.txt", 3000)
        self.format_samples("dataset_little-symbol_238753.txt", 100000)
        self.format_samples_val("dataset_little-symbol_100000.txt", 3000)
        self.format_samples("dataset_hardcase_2280.txt", 10000)
        self.format_samples_val("dataset_hardcase_10000.txt", 3000)
        self.merge_val_cases()  # 合并全部验证文件

    def process_v8(self):
        file1 = os.path.join(DATA_DIR, "english_page_1.txt")
        file2 = os.path.join(DATA_DIR, "english_page_2.txt")
        file3 = os.path.join(DATA_DIR, "english_page_3.txt")
        out_file = os.path.join(DATA_DIR, "urls_english_page.txt")
        data_line1 = read_file(file1)
        data_line2 = read_file(file2)
        data_line3 = read_file(file3)
        data_line = data_line1 + data_line2 + data_line3
        write_list_to_file(out_file, data_line)

    def process_v9(self):
        dataset_folder = os.path.join(self.out_ds_folder, "dataset_textline_195043")
        print('[Info] 文件夹: {}'.format(dataset_folder))
        out_path_format = os.path.join(self.out_files_folder, "dataset_textline_{}.txt")
        s_time = time.time()
        paths_list, names_list = traverse_dir_files(dataset_folder, is_sorted=False)
        print('[Info] 耗时: {}'.format(time.time() - s_time))
        print('[Info] 样本数: {}'.format(len(paths_list)))
        out_path = out_path_format.format(len(paths_list))
        print('[Info] 路径: {}'.format(out_path))
        create_file(out_path)
        write_list_to_file(out_path, paths_list)
        print('[Info] 写入完成: {}'.format(out_path))


def main():
    dr = DatasetReorder()
    dr.process_v7()


if __name__ == '__main__':
    main()
