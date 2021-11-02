#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 7.9.21
"""

import cv2
import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_sevices import get_vpf_service_np


class DatasetFilter(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "files_v2",
                                      "angle_dataset_all_20211026_raw", "dataset_english-page-raw_45126.txt")

    @staticmethod
    def call_service(img_bgr, service):
        try:
            res_dict = get_vpf_service_np(img_np=img_bgr, service_name=service)  # 表格
            angle = int(res_dict["data"]["angle"])
        except Exception as e:
            print('[Info] error: {}'.format(e))
            angle = 0
        return angle

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
    def process_item(data_idx, data_line):
        img_bgr = cv2.imread(data_line)
        angle1 = DatasetFilter.call_service(img_bgr, service="k8eaBD9BgC7KVz6KmiBx3F")  # v5.03
        angle2 = DatasetFilter.call_service(img_bgr, service="vM7SwdTx45k7ur2cDwsrke")  # v5.05
        angle3 = DatasetFilter.call_service(img_bgr, service="M8LRT5PUtfjdwBwp8L9eeH")  # v5.10

        if angle1 != 0 or angle2 != 0 or angle3 != 0:
            print('[Info] angle1: {}, angle2: {}, angle3: {}'.format(angle1, angle2, angle3))
            os.remove(data_line)
            print('[Info] 删除: {}'.format(data_line))

        if data_idx % 500 == 0:
            print('[Info] data_idx: {}'.format(data_idx))

    def process(self):
        data_lines = read_file(self.file_path)
        print('[Info] 处理文件: {}'.format(self.file_path))
        print('[Info] 样本数: {}'.format(len(data_lines)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            pool.apply_async(DatasetFilter.process_item, (data_idx, data_line))
        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(self.file_path))


def main():
    duc = DatasetFilter()
    duc.process()


if __name__ == '__main__':
    main()
