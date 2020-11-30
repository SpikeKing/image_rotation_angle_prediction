#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 30.11.20
"""

import os
import sys

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
            write_line(out_path, url)


def main():
    df = DatasetFilter()
    df.filter()


if __name__ == '__main__':
    main()