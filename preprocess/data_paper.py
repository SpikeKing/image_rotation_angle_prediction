#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 13.11.20
"""

import os
import cv2

import json
import sys

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myutils.project_utils import read_file, download_url_img, download_url_txt, mkdir_if_not_exist


class DataPaper(object):
    def __init__(self):
        pass

    def process_api(self):
        img_path = os.path.join(DATA_DIR, 'papers', 'api_question_list.json')
        data_lines = read_file(img_path)
        print('[Info] 文本行数: {}'.format(len(data_lines)))
        json_data = data_lines[0]
        json_dict = json.loads(json_data)
        print("[Info] json dict: {}".format(json_dict.keys()))
        data_dict = json_dict['data']
        print('[Info] data dict: {}'.format(data_dict.keys()))
        sub_data_list = data_dict['data']
        print('[Info] sub data list: {}'.format(len(sub_data_list)))
        sub_page_info = data_dict['pageinfo']
        print('[Info] sub data list: {}'.format(sub_page_info))
        for data in sub_data_list:
            print(data)
            break

    @staticmethod
    def process_page(item_dict, out_dir):
        s_id = item_dict["id"]
        img_urls_path = item_dict['img_urls']
        is_ok, urls_list = download_url_txt(img_urls_path, is_split=True)
        for u_id, url in enumerate(urls_list):
            is_ok, img_bgr = download_url_img(url)
            out_name = "{}_{}.jpg".format(s_id, u_id)
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, img_bgr)
        print('[Info] 完成: {}, {}'.format(s_id, img_urls_path))

    def process_3499(self):
        img_path = os.path.join(DATA_DIR, 'papers', 'application_3499_1024.json')
        out_dir = os.path.join(DATA_DIR, 'papers', 'application_3499_1024')

        mkdir_if_not_exist(out_dir)

        data_lines = read_file(img_path)
        print('[Info] 文本行数: {}'.format(len(data_lines)))
        json_data = data_lines[0]
        json_list = json.loads(json_data)
        print('[Info] json list: {}'.format(len(json_list)))

        pool = Pool(processes=40)
        for idx, item_dict in enumerate(json_list):
            # DataPaper.process_page(item_dict, out_dir)
            pool.apply_async(DataPaper.process_page, (item_dict, out_dir))
            # if (idx + 1) % 10 == 0:
            #     print('[Info] idx: {}'.format(idx+1))

        pool.close()
        pool.join()


def main():
    dp = DataPaper()
    dp.process_3499()


if __name__ == '__main__':
    main()
