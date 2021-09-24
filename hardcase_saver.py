#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.9.21
"""
import argparse

from multiprocessing.pool import Pool
from myutils.project_utils import *
from myutils.cv_utils import *


class HardcaseSaver(object):
    """
    存储HardCase, 与service_tester.py配合使用
    1. 验证错误case
    2. 将错误case加入hardcase文件夹
    """
    def __init__(self):
        self.dataset_path = os.path.join('..', 'datasets', 'rotation_datasets_hardcase')
        print('[Info] hardcase路径: {}'.format(self.dataset_path))
        mkdir_if_not_exist(self.dataset_path)

    @staticmethod
    def center_crop_by_hw(img_bgr):
        """
        避免图像的比例失衡
        """
        h, w, _ = img_bgr.shape
        if h // w > 3:
            mid = h // 2
            img_crop = img_bgr[mid - w:mid + w, :, :]
            return img_crop
        if w // h > 3:
            mid = w // 2
            img_crop = img_bgr[:, mid - h:mid + h, :]
            return img_crop
        else:
            return img_bgr

    @staticmethod
    def resize_max_fixed(img_bgr, size=1024):
        """
        根据最大边resize
        """
        h, w, _ = img_bgr.shape
        if h >= w:
            w = int(w * size / h)
            h = size
        else:
            h = int(h * size / w)
            w = size
        img_bgr = cv2.resize(img_bgr, (w, h))
        return img_bgr

    @staticmethod
    def process_line(data_idx, data_line, folder_path):
        items = data_line.split("\t")
        img_url = items[0]
        _, img_bgr = download_url_img(img_url)
        img_bgr = HardcaseSaver.center_crop_by_hw(img_bgr)  # 取图像中心
        img_bgr = HardcaseSaver.resize_max_fixed(img_bgr)  # 根据最长边resize
        file_name_x = img_url.split("/")[-1].split(".")[0]
        file_name = "{}-time-{}.jpg".format(file_name_x, get_current_time_str())
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, img_bgr)
        print('[Info] 处理完成: {}'.format(data_idx))

    def process(self, file_path):
        print('[Info] 处理文件路径: {}'.format(file_path))
        data_lines = read_file(file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        folder_path = os.path.join(self.dataset_path, "{}".format(get_current_day_str()))
        mkdir_if_not_exist(folder_path)

        paths_list, names_list = traverse_dir_files(self.dataset_path)
        print('[Info] 数据集样本数: {}'.format(len(names_list)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(HardcaseSaver.process_line, (data_idx, data_line, folder_path))
        pool.close()
        pool.join()

        paths_list, names_list = traverse_dir_files(self.dataset_path)
        print('[Info] 数据集样本数: {}'.format(len(names_list)))
        print('[Info] 写入完成: {}'.format(folder_path))


def parse_args():
    """
    处理脚本参数，支持相对路径
    """
    parser = argparse.ArgumentParser(description='生成hardcase')
    parser.add_argument('-f', dest='file_path', required=False, help='输入文件', type=str)

    args = parser.parse_args()

    arg_file_path = args.file_path
    print("输入文件: {}".format(arg_file_path))

    return arg_file_path


def main():
    arg_file_path = parse_args()
    hcs = HardcaseSaver()
    hcs.process(arg_file_path)


if __name__ == '__main__':
    main()
