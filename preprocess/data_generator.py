#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 10.11.20
"""

import os
import cv2

from root_dir import DATA_DIR
from myutils.cv_utils import show_img_bgr
from myutils.project_utils import traverse_dir_files, mkdir_if_not_exist


class DataGenerator(object):
    """
    通过整页的素材数据，生成子图像
    """
    def __init__(self):
        pass

    def process_img(self, img_path):
        """
        处理图像
        """
        print('[Info] img_path: {}'.format(img_path))
        img_bgr = cv2.imread(img_path)
        # show_img_bgr(img_bgr)
        h, w, _ = img_bgr.shape
        n_p = 3
        ph = h // n_p
        pw = w // n_p

        patch_list = []
        sh, sw = 0, 0
        for i in range(n_p):
            for j in range(n_p):
                img_patch = img_bgr[sh:sh+ph, sw:sw+pw, :]
                # show_img_bgr(img_patch)
                patch_list.append(img_patch)
                sw = sw + pw
            sw = 0
            sh = sh + ph
        print('[Info] patch_size: {}'.format(patch_list[0].shape))
        print('[Info] num of patch: {}'.format(len(patch_list)))
        return patch_list

    def process_folder(self, img_dir, out_dir):
        """
        处理文件夹
        :param img_dir: 输入文件夹
        :param out_dir: 输出文件夹
        :return: None
        """
        print('[Info] 处理文件夹: {}'.format(img_dir))
        print('[Info] 输出文件夹: {}'.format(out_dir))
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(img_dir)

        for path, name in zip(paths_list, names_list):
            patch_list = self.process_img(path)
            out_name_f = name.split('.')[0] + ".o{}.jpg"
            out_path_f = os.path.join(out_dir, out_name_f)
            for idx, img_p in enumerate(patch_list):
                out_path = out_path_f.format(idx)
                cv2.imwrite(out_path, img_p)

        print('[Info] 处理完成: {}'.format(out_dir))


def generate_patches():
    dg = DataGenerator()

    img_dir = os.path.join(DATA_DIR, 'task_pages')
    out_dir = os.path.join(DATA_DIR, 'task_pages_out')
    dg.process_folder(img_dir, out_dir)


def main():
    generate_patches()


if __name__ == '__main__':
    main()