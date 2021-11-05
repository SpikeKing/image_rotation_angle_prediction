#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 10.11.20
"""

import argparse
import os
from multiprocessing import Pool

import cv2


def traverse_dir_files(root_dir, ext=None):
    """
    列出文件夹中的文件, 深度遍历
    :param root_dir: 根目录
    :param ext: 后缀名
    :return: [文件路径列表, 文件名称列表]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    return paths_list, names_list


def check_img(idx, path, size):
    """
    检查图像
    """
    is_good = True
    try:
        img_bgr = cv2.imread(path)
        h, w, _ = img_bgr.shape
        if h < size or w < size:
            is_good = False
        _ = cv2.resize(img_bgr, (size, size))
    except Exception as e:
        is_good = False

    if path.endswith("jpg"):
        with open(path, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print('[Info] Not complete jpg image')
            is_good = False

    if not is_good:
        print('[Info] error path: {}'.format(path))
        os.remove(path)

    if (idx + 1) % 1000 == 0:
        print('[Info] idx: {}'.format(idx + 1))


def read_file(data_file, mode='more'):
    """
    读文件, 原文件和数据文件
    :return: 单行或数组
    """
    try:
        with open(data_file, 'r', errors='ignore') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                output = [o.strip() for o in output]
                return output
            else:
                return list()
    except IOError:
        return list()


def check_error(img_dir, n_prc, size):
    """
    检查错误图像的数量
    """
    print('[Info] 处理文件夹路径: {}'.format(img_dir))
    if img_dir.endswith(".txt"):
        paths_list = read_file(img_dir)
    else:
        paths_list, names_list = traverse_dir_files(img_dir)
    print('[Info] 数据总量: {}'.format(len(paths_list)))

    pool = Pool(processes=n_prc)  # 多线程下载
    for idx, path in enumerate(paths_list):
        pool.apply_async(check_img, (idx, path, size))

    pool.close()
    pool.join()

    print('[Info] 数据处理完成: {}'.format(img_dir))


def parse_args():
    """
    处理脚本参数，支持相对路径
    :return: in_folder 输入文件夹, size 尺寸, n_prc 进程数
    """
    parser = argparse.ArgumentParser(description='检查图片脚本')
    parser.add_argument('-i', dest='in_folder', required=True, help='输入文件夹', type=str)
    parser.add_argument('-p', dest='n_prc', required=False, default=100, help='进程数', type=str)
    parser.add_argument('-s', dest='size', required=False, default=20, help='最小边长', type=str)
    args = parser.parse_args()

    in_folder = args.in_folder
    size = int(args.size)
    n_prc = int(args.n_prc)
    print("文件路径：{}".format(in_folder))
    print("进程数: {}".format(n_prc))
    print("边长: {}".format(size))

    return in_folder, n_prc, size


def main():
    arg_in, n_prc, size = parse_args()
    check_error(arg_in, n_prc, size)


if __name__ == '__main__':
    main()