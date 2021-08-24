#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.8.21
"""

import os
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myutils.cv_utils import *
from myutils.project_utils import *
from x_utils.vpf_sevices import *


class DataProcessorV2(object):
    def __init__(self):
        self.file_name = os.path.join(DATA_DIR, "files", "10w_online_query_0823.txt")
        self.out_file_name = os.path.join(DATA_DIR, "files",
                                          "10w_online_query_0823.out-{}.txt".format(get_current_time_str()))

    @staticmethod
    def get_rotation_from_service(img_url):
        res_dict = get_vpf_service(img_url, "mJhcySi7TS3ChV6JWba4pi")
        angle = res_dict["data"]["angle"]
        return angle

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Image-Rotation/datasets/img-translation-v1"
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_line(img_idx, img_url, out_file_name):
        angle = DataProcessorV2.get_rotation_from_service(img_url)
        _, img_bgr = download_url_img(img_url)
        img_new_bgr = rotate_img_for_4angle(img_bgr, angle)
        img_new_name = "{}-angle-{}-{}.jpg".format(img_url.split("/")[-1].split(".")[0], angle, get_current_day_str())
        img_new_url = DataProcessorV2.save_img_path(img_new_bgr, img_new_name)
        write_line(out_file_name, img_new_url)
        if img_idx % 100 == 0:
            print("处理完成 idx: {}".format(img_idx))

    def process(self):
        data_lines = read_file(self.file_name)
        print('[Info] 文件: {}'.format(self.file_name))
        print('[Info] 样本数: {}'.format(len(data_lines)))
        pool = Pool(processes=100)
        for img_idx, img_url in enumerate(data_lines):
            # DataProcessorV2.process_line(img_idx, img_url, self.out_file_name)
            pool.apply_async(DataProcessorV2.process_line, (img_idx, img_url, self.out_file_name))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_file_name))


def main():
    dp2 = DataProcessorV2()
    dp2.process()


if __name__ == '__main__':
    main()

