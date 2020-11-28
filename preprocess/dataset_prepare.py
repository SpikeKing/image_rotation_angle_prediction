#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 26.11.20
"""

import os
import cv2
import json
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from x_utils.vpf_utils import get_uc_rotation_vpf_service, get_dmy_rotation_vpf_service
from x_utils.oss_utils import save_img_2_oss
from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import ROOT_DIR, DATA_DIR


class DatasetPrepare(object):
    def __init__(self):
        self.file_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_28')
        self.vpf_dir = os.path.join(DATA_DIR, '2020_11_28_out')

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
                # img_id = items[0]
                img_url = items[0].split('?')[0]
                dmy_dict = json.loads(items[1])
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
        out_dir = self.vpf_dir
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
        out_right_file = os.path.join(out_dir, '{}_right.txt'.format(name_x))
        out_diff_file = os.path.join(out_dir, '{}_diff.txt'.format(name_x))
        create_file(out_right_file)
        create_file(out_diff_file)

        out_right_list, out_diff_list = [], []
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
            out_items = [img_url, str(dmy_angle), str(uc_angle)]
            print('[Info] {} out_items: {}'.format(idx, out_items))
            out_line = ",".join(out_items)
            if dmy_angle != uc_angle:
                out_diff_list.append(out_line)
            else:
                out_right_list.append(out_line)

        write_list_to_file(out_right_file, out_right_list)
        write_list_to_file(out_diff_file, out_diff_list)
        print('[Info] 处理完成: {} {}'.format(out_right_file, out_diff_file))

    def process_vpf_data(self):
        paths_list, names_list = traverse_dir_files(self.vpf_dir)
        out_dir = os.path.join(DATA_DIR, '2020_11_28_vpf')
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
        vpf_right_path = os.path.join(DATA_DIR, '2020_11_26_right.txt')
        vpf_diff_path = os.path.join(DATA_DIR, '2020_11_26_diff.txt')
        paths_list, names_list = traverse_dir_files(vpf_dir)
        out_right_list, out_diff_list = [], []
        for path, name in zip(paths_list, names_list):
            data_lines = read_file(path)
            if name.endswith('_right.txt'):
                out_right_list += data_lines
            elif name.endswith('_diff.txt'):
                out_diff_list += data_lines
        write_list_to_file(vpf_right_path, out_right_list)
        write_list_to_file(vpf_diff_path, out_diff_list)
        print('[Info] 正确 {} 行 写入文件: {}'.format(len(out_right_list), vpf_right_path))
        print('[Info] Diff {} 行 写入文件: {}'.format(len(out_diff_list), vpf_diff_path))

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

    def generate_right_angle(self):
        urls_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_out')
        diff_urls_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_vpf.txt')
        same_urls_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_same.txt')
        data_lines = read_file(diff_urls_file)
        diff_urls = []
        for data_line in data_lines:
            items = data_line.split(',')
            url = items[0]
            diff_urls.append(url)
        print('[Info] diff_urls: {}'.format(len(diff_urls)))
        print('[Info] diff_urls sample: {}'.format(diff_urls[0]))

        all_out_list = []
        paths_list, names_list = traverse_dir_files(urls_dir)
        for path, name in zip(paths_list, names_list):
            data_lines = read_file(path)
            out_list = []
            for data_line in data_lines:
                items = data_line.split(',')
                url = items[0]
                angle = items[1]
                if url in diff_urls:
                    continue
                out_line = "{},{}".format(url, angle)
                out_list.append(out_line)
            print('[Info] {} items: {}, same: {}'.format(name, len(data_lines), len(out_list)))
            all_out_list += out_list

        print('[Info] all same: {}'.format(len(all_out_list)))
        write_list_to_file(same_urls_file, all_out_list)
        print('[Info] 处理完成: {}'.format(same_urls_file))

    @staticmethod
    def process_img_angle(idx, url, angle, out_dir):
        try:
            angle = int(angle)
            is_ok, img_bgr = download_url_img(url)
            img_out = rotate_img_for_4angle(img_bgr, angle)
            out_path = os.path.join(out_dir, "{}.jpg".format(idx))
            cv2.imwrite(out_path, img_out)
        except Exception as e:
            print('[Error] {} 错误'.format(idx))
            return
        print('[Info] {} {} 完成'.format(idx, out_path))

    def download_right_angle(self):
        same_urls_file = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_same.txt')
        out_dir = os.path.join(ROOT_DIR, '..', 'datasets', '2020_11_26_imgs_dir')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(same_urls_file)
        random.shuffle(data_lines)

        pool = Pool(processes=80)
        for idx, data_line in enumerate(data_lines):
            # if idx == 200:
            #     break
            url, angle = data_line.split(',')
            # DatasetPrepare.process_img_angle(idx, url, angle, out_dir)
            pool.apply_async(DatasetPrepare.process_img_angle, (idx, url, angle, out_dir))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(out_dir))

    def filter_diff_file(self):
        diff_path = os.path.join(DATA_DIR, '2020_11_26_vpf.txt')
        diff_right_path = os.path.join(DATA_DIR, '2020_11_26_vpf.right.txt')
        diff_error_path = os.path.join(DATA_DIR, '2020_11_26_vpf.error.txt')
        data_lines = read_file(diff_path)

        right_count, error_count = 0, 0
        for idx, data_line in enumerate(data_lines):
            img_url, old_dmy_angle, uc_angle = data_line.split(',')
            old_dmy_angle = int(old_dmy_angle)
            uc_angle = int(uc_angle)

            try:
                dmy_dict = get_dmy_rotation_vpf_service(img_url)
                dmy_angle = int(dmy_dict['data']['angel'])
                dmy_angle = format_angle(dmy_angle)
            except Exception as e:
                out_line = "{},{},{},{}".format(img_url, uc_angle, old_dmy_angle, -1)
                write_line(diff_error_path, out_line)
                error_count += 1
                print('[Info] error: {}'.format(data_line))
                continue
            if dmy_angle == old_dmy_angle or dmy_angle == uc_angle:
                out_line = "{},{}".format(img_url, dmy_angle)
                write_line(diff_right_path, out_line)
                right_count += 1
            else:
                out_line = "{},{},{},{}".format(img_url, uc_angle, old_dmy_angle, dmy_angle)
                write_line(diff_error_path, out_line)
                error_count += 1
            print('[Info] {} right: {}, error: {}'.format(img_url, right_count, error_count))


def main():
    dp = DatasetPrepare()
    # dp.process_raw_data()
    dp.process_vpf_data()
    # dp.merge_vpf_data()
    # dp.generate_labeled_data()
    # dp.generate_right_angle()
    # dp.download_right_angle()
    # dp.filter_diff_file()


if __name__ == '__main__':
    main()
