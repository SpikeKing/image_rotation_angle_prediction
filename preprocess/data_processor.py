#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 4.11.20
"""

import os
import copy
import cv2
import numpy as np

from myutils.cv_utils import get_patch
from root_dir import DATA_DIR


class DataProcessor(object):
    """
    数据处理
    """
    def __init__(self):
        pass

    def cut_img_without_margin(self, img_bgr):
        """
        去掉旋转图像四个角的黑色区域，保留最大的内接矩形
        :param img_bgr: bgr图像
        :return: 最大的内接矩形图像
        """
        from myutils.cv_utils import show_img_bgr
        show_img_bgr(img_bgr)

        h, w, _ = img_bgr.shape
        img_copy = copy.copy(img_bgr)

        # 从4个角，洪水漫灌，选择4个角的联通区域，
        mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, 0), newVal=(0, 0, 255),
                      loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, 0), newVal=(0, 0, 255),
                      loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, h-1), newVal=(0, 0, 255),
                      loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, h-1), newVal=(0, 0, 255),
                      loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
        show_img_bgr(img_copy)

        mask_inv = np.where(mask > 0.5, 0, 1).astype(np.uint8)  # 反转mask的0和1

        # 形态学Opening，过滤较小的非连通区域
        kernel = np.ones((9, 9), np.uint8)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
        show_img_bgr(mask_inv * 255)

        # 中心区域是1，其他位置是0
        contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_points = np.squeeze(contours)  # 确保1个连通区域，多个连通区域会异常

        # 测试连通区域
        cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), 3)
        show_img_bgr(img_bgr)

        # 获取点的全部X\Y坐标
        x_list, y_list = [], []
        for p in c_points:
            x_list.append(p[0])
            y_list.append(p[1])
        x_list = np.asarray(x_list)
        y_list = np.asarray(y_list)

        # 从XY坐标中，提取区域的最大矩形区域，即[x_min, y_min, x_max, y_max]
        x_min_idx = np.argmin(x_list)
        y_1 = y_list[x_min_idx]
        x_max_idx = np.argmax(x_list)
        y_2 = y_list[x_max_idx]
        y_max, y_min = max(y_1, y_2), min(y_1, y_2)

        y_min_idx = np.argmin(y_list)
        x_1 = x_list[y_min_idx]
        y_max_idx = np.argmax(y_list)
        x_2 = x_list[y_max_idx]
        x_max, x_min = max(x_1, x_2), min(x_1, x_2)

        # 获取核心的中心区域
        img_patch = get_patch(img_bgr, [x_min, y_min, x_max, y_max])
        show_img_bgr(img_patch)

        # 写入输出图像
        img_out_path = os.path.join(DATA_DIR, 'out1.jpg')
        cv2.imwrite(img_out_path, img_patch)

        print('[Info] 处理完成!')

        return img_patch


def demo_of_cut_img_without_margin():
    """
    测试cut_img_without_margin
    """
    dp = DataProcessor()

    img_path = os.path.join(DATA_DIR, '1.jpg')
    img_bgr = cv2.imread(img_path)
    dp.cut_img_without_margin(img_bgr)


def main():
    demo_of_cut_img_without_margin()


if __name__ == '__main__':
    main()