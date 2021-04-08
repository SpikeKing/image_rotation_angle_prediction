#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 8.4.21
"""

import os
import sys
import re
import pandas as pd
from myutils.project_utils import *

from root_dir import DATA_DIR


class DatasetXiaotu(object):
    def __init__(self):
        self.file_name = os.path.join(DATA_DIR, 'xiaotu_labeled_files',
                                      '2d4f6a96-1313-4f04-b3f1-255a7eb8f81c_165511.csv')
        self.out_file = os.path.join(DATA_DIR, 'xiaotu_labeled_files', 'xiaotu_labeled_25w_165512.txt')
        create_file(self.out_file)

    def process(self):
        print('[Info] file_name: {}'.format(self.file_name))
        # data_lines = read_file(self.file_name)
        df = pd.read_csv(self.file_name)
        print('[Info] 文本行数: {}'.format(len(df)))
        for idx, row in df.iterrows():
            # print(row)
            question_content = row["question_content"].replace("'", "\"")
            url = json.loads(question_content)[1]
            radio = row["radio_1"]
            # print(radio)
            # print(url)
            x = re.findall(r"(.+?)\[(.+?)\]", radio)
            label = int(x[0][0])
            label_prob = float(x[0][1])
            if label == 0 and label_prob == 1.0:
                write_line(self.out_file, url)
            else:
                print('[Info] error idx: {}'.format(idx))
            if idx % 10000 == 0:
                print('[Info] idx: {}'.format(idx))


def main():
    dx = DatasetXiaotu()
    dx.process()


if __name__ == '__main__':
    main()
