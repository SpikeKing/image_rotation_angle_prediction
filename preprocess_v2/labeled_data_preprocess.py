#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 4.9.21
"""
import os
import sys
import urllib
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_sevices import get_vpf_service


class LabeledDataPreprocess(object):
    """
    标注数据预处理
    """
    def __init__(self):
        self.files_folder = os.path.join(DATA_DIR, "nat_url_files")  # 自然场景URL汇总
        self.out_urls_file = os.path.join(DATA_DIR, "files_v2", "nat_urls.{}.txt".format(get_current_day_str()))
        self.pre_urls_file = os.path.join(DATA_DIR, "files_v2", "nat_urls.{}.prelabeled.txt".format(get_current_day_str()))

    @staticmethod
    def get_rotation_from_service(img_url):
        # qPEdfEwcvDNKAHpGCLYjBK, 自然场景
        res_dict = get_vpf_service(img_url, "qPEdfEwcvDNKAHpGCLYjBK")
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

    def process_step1(self):
        if is_file_nonempty(self.out_urls_file):
            print('[Info] 文件处理完成: {}'.format(self.out_urls_file))
            return

        paths_list, names_list = traverse_dir_files(self.files_folder)
        print('[Info] 文件夹: {}'.format(self.files_folder))
        print('[Info] 文件数: {}'.format(len(paths_list)))

        data_lines = []
        for path in paths_list:
            sub_lines = read_file(path)
            data_lines += sub_lines
        print('[Info] 文件行数: {}'.format(len(data_lines)))

        print('[Info] 输出urls: {}'.format(self.out_urls_file))

        create_file(self.out_urls_file)
        urls_list = []
        for data_line in data_lines:
            items = data_line.split(":", 1)  # 只切分两个部分，即冒号只切一次
            url = items[1]
            urls_list.append(url)
        write_list_to_file(self.out_urls_file, urls_list)
        print('[Info] 图像数: {}'.format(len(urls_list)))

    @staticmethod
    def process_item(data_idx, data_line, out_file):
        try:
            # print('[Info] img_url: {}'.format(data_line))
            img_name = data_line.split("?")[0].split("%2F")[-1].split(".")[0]
            img_name = urllib.parse.unquote(img_name)
            angle = LabeledDataPreprocess.get_rotation_from_service(data_line)
            out_img_name = "{}-x-{}-x-{}-x-angle-{}.jpg".format(data_idx, img_name, get_current_day_str(), angle)
            # print('[Info] out_img_name: {}'.format(out_img_name))
            _, img_bgr = download_url_img(data_line)
            out_img = rotate_img_for_4angle(img_bgr, angle)
            out_url = LabeledDataPreprocess.save_img_path(out_img, out_img_name)
            # print("[Info] out_url: {}".format(out_url))
            write_line(out_file, out_url)
            print('[Info] data_idx: {}, angle: {} 完成'.format(data_idx, angle))
        except Exception as e:
            print('[Error] data_idx: {} 完成'.format(data_idx))
            print('[Error] e: {} 异常'.format(e))

    def process_step2(self):
        data_lines = read_file(self.out_urls_file)
        print('[Info] 数据行数: {}'.format(len(data_lines)))
        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            LabeledDataPreprocess.process_item(data_idx, data_line, self.pre_urls_file)
            pool.apply_async(LabeledDataPreprocess.process_item, (data_idx, data_line, self.pre_urls_file))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.pre_urls_file))


def main():
    ldp = LabeledDataPreprocess()
    ldp.process_step2()


if __name__ == '__main__':
    main()
