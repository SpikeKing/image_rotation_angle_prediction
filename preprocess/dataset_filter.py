#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 30.11.20
"""

import os
import sys
import cv2
import pandas as pd
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR, ROOT_DIR


class DatasetFilter(object):
    def __init__(self):
        pass

    def filter(self):
        data_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf.right.txt')
        out_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf_right')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(data_file)
        for data_line in data_lines:
            url, angle = data_line.split(',')
            out_path = os.path.join(out_dir, 'angle_{}.txt'.format(angle))
            write_line(out_path, data_line)

    @staticmethod
    def process_img_angle(idx, url, angle, out_dir):
        try:
            angle = int(angle)
            name = url.split('/')[-1]
            is_ok, img_bgr = download_url_img(url)
            img_out = rotate_img_for_4angle(img_bgr, angle)
            out_path = os.path.join(out_dir, "{}".format(name))
            cv2.imwrite(out_path, img_out)
        except Exception as e:
            print('[Error] {} 错误'.format(idx))
            return
        print('[Info] {} {} 完成'.format(idx, out_path))

    def download_right_angle(self):
        files_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf_right')
        paths_list, names_list = traverse_dir_files(files_dir)

        pool = Pool(processes=80)
        for path, name in zip(paths_list, names_list):
            name_x = name.split('.')[0]
            urls_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf_right', '{}.txt'.format(name_x))  # 输入
            out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked', 'vpf_right', name_x)  # 输出
            mkdir_if_not_exist(out_dir)

            data_lines = read_file(urls_file)

            for idx, data_line in enumerate(data_lines):
                url, angle = data_line.split(',')
                pool.apply_async(DatasetFilter.process_img_angle, (idx, url, angle, out_dir))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(files_dir))

    def download_right_angle_v2(self):
        urls_file = os.path.join(DATA_DIR, 'test_1000_res.right.csv')  # 输入
        out_dir = os.path.join(DATA_DIR, 'test_1000_res_right')  # 输出
        mkdir_if_not_exist(out_dir)
        data_lines = read_file(urls_file)

        pool = Pool(processes=80)
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            items = data_line.split(',')
            url = items[0]
            angle = items[1]
            pool.apply_async(DatasetFilter.process_img_angle, (idx, url, angle, out_dir))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(out_dir))

    def read_labeled_data(self):
        labeled_path = os.path.join(DATA_DIR, 'labeled_20201130_v1.csv')
        out_dir = os.path.join(DATA_DIR, 'labeled_20201130_v1')
        mkdir_if_not_exist(out_dir)

        pd_head = pd.read_csv(labeled_path, encoding="gb2312", header=0)
        for idx, row in enumerate(pd_head.iterrows()):
            items = row[1]
            # print('[Info] items: {}'.format(items))
            question_content = items["question_content"]
            url = json.loads(question_content)[1]
            # print('[Info] url: {}'.format(url))
            radio_1 = items["radio_1"]
            # print('[Info] radio_1: {}'.format(radio_1))
            prob = re.findall(r'[\[](.*?)[\]]', radio_1)[0]  # 获取概率
            label = re.findall(r'(.*?)[\[]', radio_1)[0]  # 获取标签
            # print('[Info] prob: {}'.format(prob))
            # print('[Info] label: {}'.format(label))
            if label == "0" and prob == "1.0":
                out_path = os.path.join(out_dir, 'data_{}_{}.txt'.format(label, prob))
                write_line(out_path, url)
            elif label == "0" and prob == "0.6666666666666666":
                out_path = os.path.join(out_dir, 'data_{}_{}.txt'.format(label, prob))
                write_line(out_path, url)

            if idx % 1000 == 0:
                print('[Info] idx: {}'.format(idx))


def main():
    df = DatasetFilter()
    # df.filter()
    df.download_right_angle_v2()
    # df.read_labeled_data()


if __name__ == '__main__':
    main()