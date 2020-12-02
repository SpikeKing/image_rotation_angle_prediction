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
from x_utils.vpf_utils import get_uc_rotation_vpf_service
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

    def generate_checked_urls(self):
        """
        https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_rotation/datasets/
        datasets_v4_checked/checked_19881/O1CN01002cx31NZW01vj7uW_!!6000000001584-0-quark.jpg
        """
        url_format = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_rotation/datasets/" \
                     "datasets_v4_checked/{}/{}"

        out_dir = os.path.join(DATA_DIR, 'datasets_v4_checked_urls')
        mkdir_if_not_exist(out_dir)
        img_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked')
        paths_list, names_list = traverse_dir_files(img_dir, is_sorted=True)

        idx = 0
        for path, name in zip(paths_list, names_list):
            items = path.split('/')
            folder = items[-2]
            out_line = url_format.format(folder, name)
            out_path = os.path.join(out_dir, '{}.txt'.format(folder))
            write_line(out_path, out_line)
            idx += 1
            if idx % 10000 == 0:
                print('[Info] idx: {}'.format(idx))
        print('[Info] 处理完成: {}'.format(out_dir))

    @staticmethod
    def check_url(idx, url, out_error_path, out_right_path):
        # print('[Info] url: {}'.format(url))
        try:
            res_dict = get_uc_rotation_vpf_service(url)
            angle = res_dict['data']['angle']
        except Exception as e:
            print('[Info] error url: {}'.format(url))
            return
        angle = int(angle)

        if angle != 0:
            print('[Info] idx: {}, angle: {}, url: {}'.format(idx, angle, url))
            write_line(out_error_path, url+","+str(angle))
        else:
            write_line(out_right_path, url+","+str(angle))
        if idx % 1000 == 0:
            print('[Info] idx: {}'.format(idx))

    def filter_checked_urls(self):
        in_dir = os.path.join(DATA_DIR, 'datasets_v4_checked_urls')
        out_dir = os.path.join(DATA_DIR, 'datasets_v4_checked_urls_out')
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(in_dir)
        pool = Pool(processes=40)

        idx = 0
        for in_path, in_name in zip(paths_list, names_list):
            # in_path = os.path.join(DATA_DIR, 'checked_19881_urls.txt')
            out_error_path = os.path.join(out_dir, '{}.error.txt'.format(in_name))
            out_right_path = os.path.join(out_dir, '{}.right.txt'.format(in_name))
            print('[Info] out_file: {} - {}'.format(out_error_path, out_right_path))
            data_lines = read_file(in_path)
            print('[Info] 文本数量: {}'.format(len(data_lines)))
            for data_line in data_lines:
                url = data_line
                # DatasetFilter.check_url(idx, url, out_path)
                pool.apply_async(DatasetFilter.check_url, (idx, url, out_error_path, out_right_path))
                if idx % 1000 == 0:
                    print('[Info] idx: {}'.format(idx))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(out_dir))

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
        labeled_path = os.path.join(DATA_DIR, 'labeled_20201202_v1.csv')
        out_dir = os.path.join(DATA_DIR, 'labeled_20201202_v1')
        mkdir_if_not_exist(out_dir)

        # pd_head = pd.read_csv(labeled_path, encoding="gb2312", header=0)
        pd_head = pd.read_csv(labeled_path, header=0)
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
    # df.generate_checked_urls()
    df.filter_checked_urls()
    # df.read_labeled_data()


if __name__ == '__main__':
    main()