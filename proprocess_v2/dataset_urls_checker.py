#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 7.9.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_sevices import get_vpf_service


class DatasetUrlsChecker(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "files_v2", "k12_urls_english.txt")
        self.file_txt_path = os.path.join(DATA_DIR, "files_v2", "k12_urls_english.check.txt")
        self.file_html_path = os.path.join(DATA_DIR, "files_v2", "k12_urls_english.check.html")

    @staticmethod
    def call_service(img_url):
        service = "RZQqg7HkMMFf5A2CKtoVB3"
        res_dict = get_vpf_service(img_url=img_url, service_name=service)  # 表格
        angle = res_dict["data"]["angle"]
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
    def process_item(data_idx, data_line, out_path):
        angle = DatasetUrlsChecker.call_service(data_line)
        angle = int(angle)
        _, img_bgr = download_url_img(data_line)
        if angle != 0:
            print('[Info] angle: {}'.format(angle))
            print('[Info] data_line: {}'.format(data_line))
            img_bgr = rotate_img_for_4angle(img_bgr, angle)

        img_name = "{}-{}.jpg".format(data_idx, get_current_time_str())
        img_url = DatasetUrlsChecker.save_img_path(img_bgr, img_name)
        write_line(out_path, img_url)
        print('[Info] data_idx: {}'.format(data_idx))

    def process(self):
        data_lines = read_file(self.file_path)
        print('[Info] 处理文件: {}'.format(self.file_path))
        print('[Info] 样本数: {}'.format(len(data_lines)))

        random.seed(47)
        random.shuffle(data_lines)

        # data_lines = data_lines[:200]
        print('[Info] 样本数: {}'.format(len(data_lines)))

        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            DatasetUrlsChecker.process_item(data_idx, data_line, self.file_txt_path)
            # pool.apply_async(DatasetUrlsChecker.process_item, (data_idx, data_line, self.file_txt_path))
        pool.close()
        pool.join()

        urls_list = read_file(self.file_txt_path)
        print('[Info] 输出文件: {}'.format(self.file_txt_path))
        print('[Info] 样本数: {}'.format(len(urls_list)))
        urls_list = [[x] for x in urls_list]

        make_html_page(self.file_html_path, urls_list)
        print('[Info] 处理完成: {}'.format(self.file_html_path))


def main():
    duc = DatasetUrlsChecker()
    duc.process()


if __name__ == '__main__':
    main()
