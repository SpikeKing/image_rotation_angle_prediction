#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 14.9.21
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR
from train.problem_trainer import ProblemTrainer


class DatasetSaver(object):
    """
    预存数据文件，避免遍历速度过慢
    """
    def __init__(self):
        self.train_format_file = os.path.join(DATA_DIR, "files_v2", "train_dataset_all_{}.txt")
        self.test_format_file = os.path.join(DATA_DIR, "files_v2", "test_dataset_all_{}.txt")

    def process(self):
        train_filenames, test_filenames = ProblemTrainer.load_train_and_test_dataset_v1()
        print('[Info] 训练样本数: {}'.format(len(train_filenames)))
        print('[Info] 测试样本数: {}'.format(len(test_filenames)))
        train_path = self.train_format_file.format(len(train_filenames))
        test_path = self.test_format_file.format(len(test_filenames))
        if os.path.exists(train_path):
            print("[Info] 文件存在: {}".format(train_path))
            return
        if os.path.exists(test_path):
            print("[Info] 文件存在: {}".format(test_path))
            return
        write_list_to_file(train_path, train_filenames)
        write_list_to_file(test_path, test_filenames)
        print('[Info] 写入完成: {}'.format(train_path))
        print('[Info] 写入完成: {}'.format(test_path))


def main():
    ds = DatasetSaver()
    ds.process()


if __name__ == '__main__':
    main()
