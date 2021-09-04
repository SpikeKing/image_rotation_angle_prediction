#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.8.21
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myutils.project_utils import *
from x_utils.vpf_sevices import *


class DataProcessorV3(object):
    def __init__(self):
        # self.label_file = os.path.join(DATA_DIR, "files", "18c8e245-da79-4b94-9311-02914430035a_166499.csv")
        # self.relabel_file = os.path.join(DATA_DIR, "files", "02c0bb86-ebc3-4f99-8759-25178d5aba33_166505.csv")

        self.label_file = os.path.join(DATA_DIR, "files", "6d6895c2-6071-42cc-927f-a84fa6df3c10_166511.csv")
        self.relabel_file = self.label_file

        time_str = get_current_time_str()

        self.out_file_right_name = \
            os.path.join(DATA_DIR, "files", "url_0_table_right.{}.txt".format(time_str))
        self.out_file_error_name = \
            os.path.join(DATA_DIR, "files", "url_1_table_err.{}.txt".format(time_str))
        self.out_file_unknown_name = \
            os.path.join(DATA_DIR, "files", "url_2_table_unknown.{}.txt".format(time_str))

    def process(self):
        print('[Info] 标注文件: {}'.format(self.label_file))
        _, label_list = read_csv_file(self.label_file)
        print('[Info] 标注文件行数: {}'.format(len(label_list)))
        print('[Info] 重标注文件: {}'.format(self.relabel_file))
        _, relabel_list = read_csv_file(self.relabel_file)
        print('[Info] 重标注文件行数: {}'.format(len(relabel_list)))

        relabel_dict = collections.defaultdict(list)
        for row in relabel_list:
            url = json.loads(row["问题内容"])[0]
            label_str = json.loads(row["回答内容"])["radio_1"]
            relabel_dict[url].append(label_str)

        label_dict = collections.defaultdict(list)
        l_num, rl_num = 0, 0
        for row in label_list:
            url = json.loads(row["问题内容"])[0]
            if url in relabel_dict:
                label_dict[url] = relabel_dict[url]
                rl_num += 1
            else:
                label_str = json.loads(row["回答内容"])["radio_1"]
                label_dict[url].append(label_str)
                l_num += 1
        print('[Info] 标注文件: {}, 重标文件: {}'.format(l_num, rl_num))

        count_0, count_1, count_2 = 0, 0, 0
        for url in label_dict.keys():
            label_list = label_dict[url]
            if "2" in label_list:
                count_2 += 1
                write_line(self.out_file_unknown_name, "{}\t{}".format(url, "+".join(label_list)))
            elif "1" in label_list:
                count_1 += 1
                write_line(self.out_file_error_name, "{}\t{}".format(url, "+".join(label_list)))
            else:
                label_list = list(set(label_list))
                if len(label_list) == 1 and label_list[0] == "0":
                    count_0 += 1
                    write_line(self.out_file_right_name, "{}\t{}".format(url, "+".join(label_list)))
        print('[Info] 总数: {}, 正确: {}, 错误: {}, 未知: {}'.format(len(label_dict.keys()), count_0, count_1, count_2))


def main():
    dp3 = DataProcessorV3()
    dp3.process()


if __name__ == '__main__':
    main()
