#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 26.11.20
"""

import os
import json
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from x_utils.vpf_utils import get_uc_rotation_vpf_service
from x_utils.oss_utils import save_img_2_oss
from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import ROOT_DIR, DATA_DIR


class DatasetPrepare(object):
    def __init__(self):
        self.file_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26')
        self.vpf_dir = os.path.join(DATA_DIR, '2020_11_26_out')

    @staticmethod
    def process_raw_file(path, name, out_dir):
        print('[Info] path: {}'.format(path))
        data_lines = read_file(path)
        out_file = os.path.join(out_dir, '{}_out.txt'.format(name))
        create_file(out_file)

        out_list = []
        for data_line in data_lines:
            try:
                items = data_line.split('<sep>')
                img_id = items[0]
                img_url = items[1].split('?')[0]
                dmy_dict = json.loads(items[3])
                angle = int(dmy_dict["call_result"]["angle"])
            except Exception as e:
                continue
            angle = (360 - angle) % 360
            angle = format_angle(angle)
            out_items = [img_url, str(angle)]
            # print('[Info] out_items: {}'.format(out_items))
            out_line = ",".join(out_items)
            out_list.append(out_line)
        write_list_to_file(out_file, out_list)
        print('[Info] 处理完成: {}'.format(out_file))

    def process_raw_data(self):
        paths_list, names_list = traverse_dir_files(self.file_dir)
        out_dir = os.path.join(DATA_DIR, '2020_11_26_out')
        mkdir_if_not_exist(out_dir)

        pool = Pool(processes=40)

        for path, name in zip(paths_list, names_list):
            # pool.apply_async(DatasetPrepare.process_file, (path, name, out_dir))
            DatasetPrepare.process_raw_file(path, name, out_dir)

        pool.close()
        pool.join()
        print('[Info] 全部处理完成: {}'.format(out_dir))

    @staticmethod
    def process_vpf_file(path, name, out_dir):
        print('[Info] path: {}'.format(path))
        data_lines = read_file(path)
        name_x = name.split('.')[0]
        out_file = os.path.join(out_dir, '{}_vpf.txt'.format(name_x))
        create_file(out_file)

        out_list = []
        for idx, data_line in enumerate(data_lines):
            # if idx == 100:
            #     break
            img_url, dmy_angle = data_line.split(',')
            dmy_angle = int(dmy_angle)
            try:
                uc_dict = get_uc_rotation_vpf_service(img_url)
                uc_angle = int(uc_dict['data']['angle'])
            except Exception as e:
                uc_angle = -1
            if dmy_angle != uc_angle:
                out_items = [img_url, str(dmy_angle), str(uc_angle)]
                print('[Info] {} out_items: {}'.format(idx, out_items))
            else:
                continue
            out_line = ",".join(out_items)
            out_list.append(out_line)
        write_list_to_file(out_file, out_list)
        print('[Info] 处理完成: {}'.format(out_file))

    def process_vpf_data(self):
        paths_list, names_list = traverse_dir_files(self.vpf_dir)
        out_dir = os.path.join(DATA_DIR, '2020_11_26_vpf')
        mkdir_if_not_exist(out_dir)

        pool = Pool(processes=40)

        for path, name in zip(paths_list, names_list):
            pool.apply_async(DatasetPrepare.process_vpf_file, (path, name, out_dir))
            # DatasetPrepare.process_vpf_file(path, name, out_dir)

        pool.close()
        pool.join()

        print('[Info] 全部处理完成: {}'.format(out_dir))

    def merge_vpf_data(self):
        vpf_dir = os.path.join(DATA_DIR, '2020_11_26_vpf')
        vpf_path = os.path.join(DATA_DIR, '2020_11_26_vpf.txt')
        paths_list, names_list = traverse_dir_files(vpf_dir)
        out_list = []
        for path, name in zip(paths_list, names_list):
            data_lines = read_file(path)
            out_list += data_lines
        write_list_to_file(vpf_path, out_list)
        print('[Info] {} 行 写入文件: {}'.format(len(out_list), vpf_path))

    @staticmethod
    def process_data_line(data_line, idx):
        try:
            img_url, dmy_angle, uc_angle = data_line.split(',')
            # print('[Info] img_url: {}, uc_angle: {}, dmy_angle: {}'.format(img_url, uc_angle, dmy_angle))
            is_ok, img_bgr = download_url_img(img_url)
            show_img_bgr(img_bgr)
            out_uc_name = "{}_uc.jpg".format(idx)
            uc_img = rotate_img_for_4angle(img_bgr, uc_angle)
            save_img_2_oss(uc_img, out_uc_name, "zhengsheng.wcl/problems_rotation/datasets/prelabeled_diff_20201127/")
            out_dmy_name = "{}_dmy.jpg".format(idx)
            dmy_img = rotate_img_for_4angle(img_bgr, dmy_angle)
            save_img_2_oss(dmy_img, out_dmy_name, "zhengsheng.wcl/problems_rotation/datasets/prelabeled_diff_20201127/")
        except Exception as e:
            print('[Exception] data_line: {}'.format(data_line))
            return
        print('[Info] 处理完成: {}'.format(idx))

    def generate_labeled_data(self):
        vpf_path = os.path.join(DATA_DIR, '2020_11_26_vpf.txt')
        data_lines = read_file(vpf_path)
        pool = Pool(processes=80)
        for idx, data_line in enumerate(data_lines):
            # DatasetPrepare.process_data_line(data_line, idx)
            pool.apply_async(DatasetPrepare.process_data_line, (data_line, idx))
        pool.close()
        pool.join()
        print('[Info] 全部处理完成: {}'.format(vpf_path))



def main():
    dp = DatasetPrepare()
    # dp.process_vpf_data()
    # dp.merge_vpf_data()
    dp.generate_labeled_data()


if __name__ == '__main__':
    main()
