#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 4.11.20
"""

import json
import os
import sys
from multiprocessing.pool import Pool

import cv2
import pandas as pd

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import mkdir_if_not_exist, download_url_img, traverse_dir_files
from root_dir import DATA_DIR


class DataDownloader(object):
    """
    数据下载
    """

    def __init__(self):
        pass

    @staticmethod
    def get_boxes(boxes_str):
        """
        解析数据, 获取boxes列表
        """
        bbox_list = []
        box_list = json.loads(boxes_str)
        boxes = box_list[0]

        n_box = len(boxes)
        if n_box == 0:
            return False, bbox_list

        # 示例: {"coord":[54,21,1742,513]}
        for box_dict in boxes:
            bbox = box_dict["coord"]
            bbox_list.append(bbox)

        return True, bbox_list

    @staticmethod
    def output_img(out_dir, maker, idx, label_str, url):
        """
        根据是否包含box写入文件
        """
        is_box, box_list = DataDownloader.get_boxes(label_str)  # 是否写入box

        if not is_box:
            return 0

        is_img, img_bgr = download_url_img(url)  # 下载图像
        if not is_img:
            return 0

        out_name = "{}_{}_{}.jpg".format(idx, maker, len(box_list))
        out_path = os.path.join(out_dir, out_name)

        cv2.imwrite(out_path, img_bgr)
        print('[Info] 写入文件: {}'.format(out_path))

    @staticmethod
    def process_excel(data_path, out_dir, pool):
        """
        处理excel文件
        """
        # 标签: ["index", "markTime", "checkLabel", "label", "checker", "url", "marker", "_id", "taskId"]
        pd_data = pd.read_csv(
            data_path,
            header=0,
            names=["index", "markTime", "checkLabel", "label", "checker", "url", "marker", "_id", "taskId"]
        )

        labels = pd_data["label"]
        urls = pd_data["url"]
        indexes = pd_data["index"]
        makers = pd_data["marker"]

        for idx, (p_idx, maker, label_str, url) in enumerate(zip(indexes, makers, labels, urls)):
            # 多进程存储图像
            pool.apply_async(DataDownloader.output_img, args=(out_dir, p_idx, maker, label_str, url, True))
            if idx % 1000 == 0:
                print('[Info] idx: {}'.format(idx))
                break  # 测试

        # 多进程逻辑
        pool.close()
        pool.join()

        print('[Info] 图像下载完成: {}'.format(data_path))

    def process_folder(self, task_folder, out_dir):
        """
        处理文件夹
        """
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(task_folder)
        n_prc = 40
        pool = Pool(processes=n_prc)  # 多线程下载

        for path, name in zip(paths_list, names_list):
            self.process_excel(path, out_dir, pool)  # 处理excel文件

        print('[Info] 全部下载完成')


def main():
    task_folder = os.path.join(DATA_DIR, 'task_files')
    out_dir = os.path.join(DATA_DIR, 'task-raw-out')
    dd = DataDownloader()
    dd.process_folder(task_folder, out_dir)


if __name__ == '__main__':
    main()