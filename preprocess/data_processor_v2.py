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
        # v1
        self.file_name = os.path.join(DATA_DIR, "files", "dump_table_no4_3yue.txt")
        self.out_file_name = os.path.join(DATA_DIR, "files", "dump_table_no4_3yue.out-{}.txt".format(get_current_time_str()))

        # v2
        # self.file_name = os.path.join(DATA_DIR, "files", "general_detect_url_watermark_1w_3w_dump.txt")  # v2
        # self.out_file_name = os.path.join(
        #     DATA_DIR, "files",  "general_detect_url_watermark_1w_3w_dump.out-{}.txt".format(get_current_time_str()))

        # 验证数据
        # self.file1_name = os.path.join(DATA_DIR, "files", "0cda042c-91df-436f-a97d-777219ce0cd6_166501.csv")
        # self.file2_name = os.path.join(DATA_DIR, "files", "4ba5fa95-4148-4379-b9b9-b700b7c22e2c_166501.csv")
        # time_str = get_current_time_str()

        # self.out_file_right_name = os.path.join(DATA_DIR, "files", "url_0_nat_right.{}.txt".format(time_str))
        # self.out_file_error_name = os.path.join(DATA_DIR, "files", "url_1_nat_err.{}.txt".format(time_str))
        # self.out_file_unknown_name = os.path.join(DATA_DIR, "files", "url_2_nat_unknown.{}.txt".format(time_str))


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
            DataProcessorV2.process_line(img_idx, img_url, self.out_file_name)
            # pool.apply_async(DataProcessorV2.process_line, (img_idx, img_url, self.out_file_name))
            break
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_file_name))

    def process_v2(self):
        data_lines = read_file(self.file_name)
        print('[Info] 文件: {}'.format(self.file_name))
        print('[Info] 样本数: {}'.format(len(data_lines)))
        pool = Pool(processes=100)
        for img_idx, item_data in enumerate(data_lines):
            # print(item_data)
            data_dict = json.loads(item_data.replace("'", "\""))
            img_url = data_dict["url"]
            pool.apply_async(DataProcessorV2.process_line, (img_idx, img_url, self.out_file_name))
        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_file_name))

    def process_data(self):
        print('[Info] 处理文件: {}'.format(self.file1_name))
        column_names, row_list = read_csv_file(self.file1_name)
        print('[Info] column_names: {}'.format(column_names))
        print('[Info] row_list: {}'.format(len(row_list)))
        url_dict = collections.defaultdict(list)
        for row in row_list:
            url = json.loads(row["问题内容"])[0]
            label_str = json.loads(row["回答内容"])["radio_1"]
            url_dict[url].append(label_str)
        print('[Info] 图像数: {}'.format(len(url_dict.keys())))

        for url in url_dict.keys():
            label_list = url_dict[url]
            if "2" in label_list:
                write_line(self.out_file_unknown_name, "{}\t{}".format(url, "+".join(label_list)))
            elif "1" in label_list:
                write_line(self.out_file_error_name, "{}\t{}".format(url, "+".join(label_list)))
            else:
                label_list = list(set(label_list))
                if len(label_list) == 1 and label_list[0] == "0":
                    write_line(self.out_file_right_name, "{}\t{}".format(url, "+".join(label_list)))
        print('[Info] 写入完成! {}'.format(self.out_file_right_name))


    def spilt_url(self):
        data_lines = read_file(os.path.join(DATA_DIR, "files", "url_2.txt"))
        out_file = os.path.join(DATA_DIR, "files", "url_2_x.txt")
        for data_line in data_lines:
            url = data_line.split("\t")[0]
            write_line(out_file, url)


def main():
    dp2 = DataProcessorV2()
    dp2.process_v2()


if __name__ == '__main__':
    main()

