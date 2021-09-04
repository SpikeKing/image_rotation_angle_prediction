#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 4.11.20
"""

import copy
import os
import sys

import cv2
import numpy as np
import random

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import get_patch, show_img_bgr
from myutils.project_utils import traverse_dir_files, mkdir_if_not_exist, safe_div, shuffle_two_list
from root_dir import DATA_DIR, ROOT_DIR


class DataProcessor(object):
    """
    数据预处理，预处理之后用于训练
    """
    def __init__(self):
        pass

    @staticmethod
    def is_close_color(std_color, target_color):
        x = np.abs(np.sum(np.array(target_color) - np.array(std_color)))
        print(x)
        if x < 5:
            return True
        else:
            return False

    @staticmethod
    def cut_img_without_margin(img_bgr, is_show=False):
        """
        去掉旋转图像四个角的黑色区域，保留最大的内接矩形
        :param img_bgr: bgr图像
        :param is_show: 是否显示测试图像
        :return: 最大的内接矩形图像
        """
        if is_show:
            show_img_bgr(img_bgr)

        h, w, _ = img_bgr.shape
        img_copy = copy.copy(img_bgr)

        margin_color = [0, 0, 0]
        diff = (2, 2, 2, 2)

        print('[Info] 四个角的颜色: {}, {}, {}, {}'.format(
            img_copy[0][0], img_copy[-1][0], img_copy[0][-1], img_copy[-1][-1]))

        # 从4个角，洪水漫灌，选择4个角的联通区域，
        mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask
        img_copy[0][0] = np.array(margin_color, dtype=np.uint8)
        img_copy[-1][0] = np.array(margin_color, dtype=np.uint8)
        img_copy[0][-1] = np.array(margin_color, dtype=np.uint8)
        img_copy[-1][-1] = np.array(margin_color, dtype=np.uint8)

        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, 0), newVal=(0, 0, 255),
                      loDiff=diff, upDiff=diff)
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, 0), newVal=(0, 0, 255),
                      loDiff=diff, upDiff=diff)
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, h-1), newVal=(0, 0, 255),
                      loDiff=diff, upDiff=diff)
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, h-1), newVal=(0, 0, 255),
                      loDiff=diff, upDiff=diff)
        if is_show:
            show_img_bgr(img_copy)

        mask_inv = np.where(mask > 0.5, 0, 1).astype(np.uint8)  # 反转mask的0和1

        # 形态学Opening，过滤较小的非连通区域
        kernel = np.ones((9, 9), np.uint8)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
        if is_show:
            show_img_bgr(mask_inv * 255)

        # 中心区域是1，其他位置是0
        contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_points = np.squeeze(contours)  # 确保1个连通区域，多个连通区域会异常

        # 测试连通区域
        if is_show:
            cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 3)
            show_img_bgr(img_bgr)

        # 获取点的全部X\Y坐标
        x_list, y_list = [], []
        for p in c_points:
            x_list.append(p[0])
            y_list.append(p[1])

        if not x_list or not y_list:
            return False, img_bgr

        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)

        # 从XY坐标中，提取区域的最大矩形区域，即[x_min, y_min, x_max, y_max]
        x_min_idx = np.argmin(x_list) - 1
        y_1 = y_list[x_min_idx]
        # x_max_idx = len(x_list) - np.argmax(x_list[::-1]) - 1
        x_max_idx = np.argmax(x_list) - 1
        y_2 = y_list[x_max_idx]
        y_max, y_min = max(y_1, y_2), min(y_1, y_2)

        # y_min_idx = len(y_list) - np.argmin(y_list[::-1]) - 1
        y_min_idx = np.argmin(y_list) - 1
        x_1 = x_list[y_min_idx]
        y_max_idx = np.argmax(y_list) - 1
        x_2 = x_list[y_max_idx]
        x_max, x_min = max(x_1, x_2), min(x_1, x_2)

        # 获取核心的中心区域
        img_patch = get_patch(img_bgr, [x_min, y_min, x_max, y_max])
        if is_show:
            show_img_bgr(img_patch)

        # 写入输出图像
        img_out_path = os.path.join(DATA_DIR, 'out1.jpg')
        cv2.imwrite(img_out_path, img_patch)

        print('[Info] 处理完成!')

        return True, img_patch

    @staticmethod
    def cut_img_v2(img_bgr):
        h, w, _ = img_bgr.shape
        # show_img_bgr(img_bgr)

        img_copy = copy.copy(img_bgr)
        margin_color = [0, 0, 0]

        ic1 = tuple(img_copy[0][0]) == tuple(np.array(margin_color, dtype=np.uint8))
        ic2 = tuple(img_copy[-1][0]) == tuple(np.array(margin_color, dtype=np.uint8))
        ic3 = tuple(img_copy[0][-1]) == tuple(np.array(margin_color, dtype=np.uint8))
        ic4 = tuple(img_copy[-1][-1]) == tuple(np.array(margin_color, dtype=np.uint8))

        if (ic1 or ic4) or (ic2 or ic3):
            h, w, _ = img_bgr.shape

            h_offset = safe_div(25, h)
            w_offset = safe_div(25, w)
            # print(h_offset, w_offset)

            sh = int(h_offset * h)
            eh = int((1 - h_offset) * h)
            sw = int(w_offset * w)
            ew = int((1 - w_offset) * w)
            img_patch = img_bgr[sh:eh, sw:ew, :]
        else:
            img_patch = img_bgr
        # show_img_bgr(img_patch)
        return img_patch

    @staticmethod
    def process_img(path, name, out_dir):
        out_name = name.replace('.jpg', "") + ".p.jpg"
        out_path = os.path.join(out_dir, out_name)

        try:
            img_bgr = cv2.imread(path)
            # print('[Info] path: {}'.format(path))
            # print('[Info] img_bgr: {}'.format(img_bgr.shape))
            img_patch = DataProcessor.cut_img_v2(img_bgr)
            cv2.imwrite(out_path, img_patch)
        except Exception as e:
            print('[Exception] out_path: {}'.format(out_path))

    def process_folder(self, in_dir, out_dir):
        """
        处理文件夹
        """
        print('[Info] in_dir: {}'.format(in_dir))
        print('[Info] out_dir: {}'.format(out_dir))
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(in_dir)
        print('[Info] 待处理文件数量: {}'.format(len(paths_list)))

        random.seed(47)
        paths_list, names_list = shuffle_two_list(paths_list, names_list)

        n_prc = 40
        pool = Pool(processes=n_prc)  # 多线程下载

        for idx, (path, name) in enumerate(zip(paths_list, names_list)):
            pool.apply_async(DataProcessor.process_img, args=(path, name, out_dir))
            # DataProcessor.process_img(path, name, out_dir)
            if (idx + 1) % 1000 == 0:
                print('[Info] num: {}'.format(idx + 1))

        # 多进程逻辑
        pool.close()
        pool.join()

        print('[Info] 处理完成! {}'.format(out_dir))
        return

    def filter_folder(self, in_dir):
        paths_list, names_list = traverse_dir_files(in_dir)
        print('[Info] 样本数: {}'.format(len(paths_list)))

        n_remove = 0
        count = 0
        for path, name in zip(paths_list, names_list):
            img_bgr = cv2.imread(path)
            h, w, _ = img_bgr.shape
            x = safe_div(h, w)

            if x > 2:
                print('[Info] 删除: {}'.format(path))
                os.remove(path)
                n_remove += 1
            count += 1
            if count % 100 == 0:
                print(count)

        print('[Info] 删除: {}'.format(n_remove))
        paths_list, names_list = traverse_dir_files(in_dir)
        print('[Info] 处理后, 样本数: {}'.format(len(paths_list)))



def demo_of_cut_img_without_margin():
    """
    测试cut_img_without_margin
    """
    dp = DataProcessor()

    img_path = os.path.join(DATA_DIR, '4.jpg')
    img_bgr = cv2.imread(img_path)
    # dp.cut_img_without_margin(img_bgr, True)
    dp.cut_img_v2(img_bgr)


def demo_of_process_folder():
    """
    预处理数据
    """
    dp = DataProcessor()

    data_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_13w_512')
    out_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_13w_512_p')

    dp.process_folder(data_dir, out_dir)


def demo_of_remove_folder():
    in_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_13w_512_p_x')

    dp = DataProcessor()
    dp.filter_folder(in_dir)


def main():
    # demo_of_cut_img_without_margin()
    # demo_of_process_folder()
    demo_of_remove_folder()


if __name__ == '__main__':
    main()