#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 6.9.21
"""

import os
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import ROOT_DIR, DATA_DIR
from x_utils.vpf_sevices import get_vpf_service_np
from myutils.project_utils import *
from myutils.cv_utils import *


class DatasetErrorChecker(object):
    def __init__(self):
        self.out_img_paths_file = os.path.join(DATA_DIR, "files_v2", "dataset_all_path_{}.txt".format(get_current_time_str()))
        self.out_img_paths_file_x = os.path.join(DATA_DIR, "files_v2", "dataset_all_path_20210906.txt")
        self.out_error_paths_file = os.path.join(DATA_DIR, "files_v2", "dataset_all_path_20210906_error_{}.txt".format(get_current_time_str()))

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

        pool = Pool(processes=100)
        for folder_path in folder_path_list:
            # DatasetErrorChecker.process_line(folder_path, self.out_img_paths_file)
            pool.apply_async(DatasetErrorChecker.process_line, (folder_path, self.out_img_paths_file))
        pool.close()
        pool.join()
        print('[Info] 写入完成: {}'.format(self.out_img_paths_file))

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Image-Rotation/imgs-tmp/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_img_path(img_idx, img_path, service,  out_file):
        img_bgr = cv2.imread(img_path)
        res_dict = get_vpf_service_np(img_np=img_bgr, service_name=service)  # 表格
        angle = res_dict["data"]["angle"]
        angle = int(angle)
        if angle != 0:
            items = img_path.split("/")
            img_name = "{}-f-{}.jpg".format(items[-2], items[-1])
            img_url = DatasetErrorChecker.save_img_path(img_bgr, img_name)
            write_line(out_file, "{}\t{}".format(img_url, angle))
            print('[Info] 处理完成: {}, angle: {}'.format(img_idx, angle))

    def check(self):
        data_lines = read_file(self.out_img_paths_file_x)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        service = "eR8qKVKScPRgWWfJQCDmXo"
        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(DatasetErrorChecker.process_img_path,
                             (data_idx, data_line, service, self.out_error_paths_file))
        pool.close()
        pool.join()
        data_lines = read_file(self.out_error_paths_file)
        print('[Info] 处理完成: {}, 样本数: {}'.format(self.out_error_paths_file, len(data_lines)))


def main():
    dc = DatasetErrorChecker()
    dc.check()


if __name__ == '__main__':
    main()
