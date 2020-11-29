#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 28.11.20
"""

import cv2
from myutils.project_utils import *
from myutils.cv_utils import *
from preprocess.make_image_page import get_image_paths
from root_dir import ROOT_DIR, DATA_DIR


def move_image_paths(image_dir, out_dir):
    img_paths = []
    post_fixes = ['.jpg', '.png', '.jpeg', '.bmp', '.webp', '.tif']
    mkdir_if_not_exist(out_dir)
    for filename in os.listdir(image_dir):
        filepath = os.path.join(image_dir, filename)
        for post_fix in post_fixes:
            if filepath.lower().endswith(post_fix):
                img_paths.append(filepath)
        if os.path.isdir(filepath):
            img_paths.extend(get_image_paths(filepath, post_fixes))

    img_paths = img_paths[5000:]
    for img_path in img_paths:
        name = img_path.split('/')[-1]
        out_path = os.path.join(out_dir, name)
        shutil.move(img_path, out_path)

    print('[Info] 处理完成!')

def rotate_image():
    img_dir = "/Users/wang/Desktop/3_error"
    out_dir = "/Users/wang/Desktop/3_error_r"
    mkdir_if_not_exist(out_dir)
    paths_list, names_list = traverse_dir_files(img_dir)
    for path, name in zip(paths_list, names_list):
        img_bgr = cv2.imread(path)
        img_r = rotate_img_for_4angle(img_bgr, 270)
        out_path = os.path.join(out_dir, name)
        cv2.imwrite(out_path, img_r)


def process():
    # re_right_dir = "/Users/wang/Desktop/标注数据/re_right"
    re_right_out_txt = "/Users/wang/Desktop/标注数据/re_right_out.txt"
    # paths_list, names_list = traverse_dir_files(re_right_dir)
    # for path, name in zip(paths_list, names_list):
    #     write_line(re_right_out_txt, name)
    # print('[Info] 处理完成!')

    # no_sense_dir = "/Users/wang/Desktop/标注数据/no_sense"
    no_sense_out_txt = "/Users/wang/Desktop/标注数据/no_sense_out.txt"
    # paths_list, names_list = traverse_dir_files(no_sense_dir)
    # for path, name in zip(paths_list, names_list):
    #     write_line(no_sense_out_txt, name)
    # print('[Info] 处理完成!')

    re_right_lines = read_file(re_right_out_txt)
    no_sense_lines = read_file(no_sense_out_txt)

    x_lines = re_right_lines + no_sense_lines

    right_dir = "/Users/wang/Desktop/标注数据/right"
    right_out_dir = "/Users/wang/Desktop/标注数据/right_out.txt"
    paths_list, names_list = traverse_dir_files(right_dir)
    for path, name in zip(paths_list, names_list):
        if name not in x_lines:
            write_line(right_out_dir, name)
        else:
            print(name)
    print('[Info] 处理完成!')


def process_v2():
    in_path = os.path.join(DATA_DIR, '2020_11_26_same.txt')
    right_out_txt = "/Users/wang/Desktop/标注数据/right_out_18293.txt"
    re_right_out_txt = "/Users/wang/Desktop/标注数据/re_right_out_1588.txt"
    no_sense_out_txt = "/Users/wang/Desktop/标注数据/no_sense_out_52.txt"

    raw_path = os.path.join(DATA_DIR, 'datasets_4_new', 'raw_20201126.txt')
    checked_path = os.path.join(DATA_DIR, 'datasets_4_new', 'checked_20201126.txt')
    modified_path = os.path.join(DATA_DIR, 'datasets_4_new', 'modified_20201126.txt')
    nosense_path = os.path.join(DATA_DIR, 'datasets_4_new', 'nosense_20201126.txt')

    right_names = read_file(right_out_txt)
    re_right_names = read_file(re_right_out_txt)
    no_sense_names = read_file(no_sense_out_txt)

    data_lines = read_file(in_path)
    out_data_lines = []
    for idx, data_line in enumerate(data_lines):
        url, angle_str = data_line.split(',')
        name = url.split('/')[-1]
        if name in re_right_names:
            out_line = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/" \
                       "problems_rotation/datasets/datasets_v4_modified/{}".format(name)
            write_line(modified_path, out_line)
        elif name in no_sense_names:
            write_line(nosense_path, data_line)
        elif name in right_names:
            write_line(checked_path, data_line)
        else:
            write_line(raw_path, data_line)
        if idx % 1000 == 0:
            print(idx)

    print('[Info] 处理完成')




def main():
    # image_dir = "/Users/wang/Desktop/2_7000"
    # out_dir = "/Users/wang/Desktop/2_12000"
    # move_image_paths(image_dir, out_dir)
    # rotate_image()
    process_v2()


if __name__ == '__main__':
    main()