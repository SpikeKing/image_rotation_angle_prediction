#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 20.10.21
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
from x_utils.vpf_sevices import get_vpf_service, get_vpf_service_np


class RoiChecker(object):
    def __init__(self):
        pass

    @staticmethod
    def call_roi_service(img_url):
        res_dict = get_vpf_service(img_url, service_name="FptYjaskomHTG6pF8eUvNT", param_dict={"filter_oss": "1"})
        # print('[Info] res_dict: {}'.format(res_dict))
        detect_result_debug = res_dict['data']["detect_result"]
        box = detect_result_debug[0][0]
        # print('[Info] box: {}'.format(box))
        return box

    @staticmethod
    def call_angle_roi_service(img_url, box):
        res_dict = get_vpf_service(img_url, service_name="eR8qKVKScPRgWWfJQCDmXo", param_dict={"rect": json.dumps(box)})
        # print('[Info] res_dict: {}'.format(res_dict))
        angle = res_dict['data']["angle"]
        image_roi_url = res_dict['data']["image_roi_url"]
        # print('[Info] angle: {}'.format(angle))
        return angle, image_roi_url

    @staticmethod
    def call_angle_service(img_url):
        res_dict = get_vpf_service(img_url, service_name="M8LRT5PUtfjdwBwp8L9eeH")
        # print('[Info] res_dict: {}'.format(res_dict))
        angle = res_dict['data']["angle"]
        # print('[Info] angle: {}'.format(angle))
        return angle

    @staticmethod
    def process_line(img_idx, img_url, out_file_path):
        try:
            box = RoiChecker.call_roi_service(img_url)
            angle_roi, image_roi_url = RoiChecker.call_angle_roi_service(img_url, box)
            angle = RoiChecker.call_angle_service(img_url)
            if angle_roi != angle:
                print('[Info] {}, {}, {}, {}, {}'.format(img_idx, img_url, image_roi_url, angle, angle_roi))
                out_line = "\t".join([img_url, image_roi_url, str(angle), str(angle_roi)])
                write_line(out_file_path, out_line)
        except Exception as e:
            print('[Info] Exception: {}'.format(e))
        if img_idx % 1000 == 0:
            print('[Info] img_idx: {}'.format(img_idx))
        return

    def process(self):
        # img_url = "https://113633.oss-cn-hangzhou-zmf.aliyuncs.com/IMG_20210929_100042.jpg"
        file_path = os.path.join(DATA_DIR, "nat_dataset_urls_20211020.txt")
        print("[Info] 输入文件: {}".format(file_path))
        out_file_path = os.path.join(DATA_DIR, "nat_dataset_urls_20211020.out.{}.txt".format(get_current_time_str()))
        out_html_path = os.path.join(DATA_DIR, "nat_dataset_urls_20211020.out.html")
        data_lines = read_file(file_path)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        random.seed(47)
        random.shuffle(data_lines)
        data_lines = data_lines[:1000]
        pool = Pool(processes=100)
        for data_idx, data_line in enumerate(data_lines):
            # RoiChecker.process_line(data_idx, data_line, out_file_path)
            pool.apply_async(RoiChecker.process_line, (data_idx, data_line, out_file_path))
        pool.close()
        pool.join()
        print('[Info] 写入完成: {}'.format(out_file_path))

        data_lines = read_file(out_file_path)
        items_list = []
        for data_line in data_lines:
            items_list.append(data_line.split("\t"))
        make_html_page(out_html_path, items_list)
        print('[Info] 写入完成: {}'.format(out_html_path))


def main():
    rc = RoiChecker()
    rc.process()


if __name__ == '__main__':
    main()
