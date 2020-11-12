#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 12.11.20
"""
import json
import os
import sys

from myutils.img_checker import traverse_dir_files
from myutils.project_utils import read_file, write_list_to_file
from root_dir import DATA_DIR


class GtProcess(object):
    def __init__(self):
        pass

    def process_raw_data(self, path):
        file_name = path.split('/')[-1]

        print('[Info] file_name: {}'.format(file_name))
        data_lines = read_file(path)

        res_list = []
        for data_line in data_lines:
            try:
                item_list = data_line.split('<sep>')
                # print('[Info] num of items: {}'.format(len(item_list)))
                item_id = item_list[0]
                url = item_list[1]
                ocr_json = item_list[2]
                ocr_dict = json.loads(ocr_json)
                angle = int(ocr_dict['angel'])
                out_json_dict = {
                    "id": item_id,
                    "url": url,
                    "angle": angle
                }
                res_list.append(json.dumps(out_json_dict))
            except Exception as e:
                continue

        print('[Info] 样本数量: {}'.format(len(res_list)))
        return res_list

    def process(self):
        folder_name = "2020_11_12"
        file_folder = os.path.join(DATA_DIR, folder_name)
        out_format = os.path.join(DATA_DIR, folder_name + "_out.{}.txt")

        paths_list, names_list = traverse_dir_files(file_folder)

        data_list = []
        for path, name in zip(paths_list, names_list):
            sub_list = self.process_raw_data(path)  # 根据宽高筛选图像
            data_list += sub_list
        print('[Info] 样本数: {}'.format(len(data_list)))

        out_file = out_format.format(len(data_list))
        write_list_to_file(out_file, data_list)
        print('[Info] 写入文件完成! {}'.format(out_file))


def main():
    gp = GtProcess()
    gp.process()


if __name__ == '__main__':
    main()