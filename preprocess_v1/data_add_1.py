#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 20.11.20
"""

import os
import sys
import json

from myutils.project_utils import traverse_dir_files, read_file, mkdir_if_not_exist, write_line, write_list_to_file
from root_dir import DATA_DIR


class DataAdd1(object):
    def __init__(self):
        pass

    def process_path_1(self, path, out_path):
        data_lines = read_file(path)
        """
        Task 61786 d7af1d07-4833-402d-b988-98277e997d51.csv:
        index;markTime;checkLabel;label;checker;url;marker;_id;taskId
        """
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                print(data_line)
            else:
                try:
                    items = data_line.split(';')
                    label_str = items[3]
                    url = items[5]
                    # print('[Info] label: {}, url: {}'.format(label_str, url))
                    label_list = json.loads(label_str)[0]
                    n_label = len(label_list)
                    # print('[Info] num: {}'.format(n_label))
                    if n_label > 0:
                        write_line(out_path, url)
                except Exception as e:
                    continue
                if idx % 1000 == 0:
                    print('[Info] idx: {}'.format(idx))

    def process(self):
        data_dir = os.path.join(DATA_DIR, 'biaozhu_csv')
        out_dir = os.path.join(DATA_DIR, 'biaozhu_csv_out')
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(data_dir)
        for path, name in zip(paths_list, names_list):
            print('[Info] path: {}'.format(path))
            name_items = name.split(' ')
            out_name = "_".join(name_items[0:2])
            out_path = os.path.join(out_dir, '{}.txt'.format(out_name))
            self.process_path_1(path, out_path)

    def merge_files(self):
        data_dir = os.path.join(DATA_DIR, 'biaozhu_csv_out')
        paths_list, names_list = traverse_dir_files(data_dir)
        out_path = os.path.join(DATA_DIR, 'biaozhu_csv_out.txt')

        all_data_lines = []
        for path, name in zip(paths_list, names_list):
            data_lines = read_file(path)

            for data_line in data_lines:
                data_line.replace("\"", "")
                all_data_lines.append(data_line)
        write_list_to_file(out_path, all_data_lines)


def main():
    da = DataAdd1()
    # da.process()
    da.merge_files()


if __name__ == '__main__':
    main()