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

from myutils.project_utils import *
from root_dir import DATA_DIR


class DataDownloader(object):
    """
    下载需要预处理的图像数据
    """

    def __init__(self):
        pass

    @staticmethod
    def get_boxes(boxes_str):
        """
        解析数据, 获取boxes列表
        """
        bbox_list = []
        try:
            box_list = json.loads(boxes_str)
        except Exception as e:
            print('[Exception] boxes_str: {}'.format(boxes_str))
            return False, bbox_list

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
    def process_excel(data_path, out_dir):
        """
        处理excel文件
        """
        # 标签: ["index", "markTime", "checkLabel", "label", "checker", "url", "marker", "_id", "taskId"]
        # 不指定names, 指定header，则数据以header的文本为keys
        pd_data = pd.read_csv(
            data_path,
            header=0
            # names=["index", "markTime", "checkLabel", "label", "checker", "url", "marker", "_id", "taskId"]
        )

        labels = pd_data["label"]
        urls = pd_data["url"]
        indexes = pd_data["index"]
        makers = pd_data["marker"]

        n_prc = 40
        pool = Pool(processes=n_prc)  # 多线程下载

        for idx, (p_idx, maker, label_str, url) in enumerate(zip(indexes, makers, labels, urls)):
            # 多进程存储图像
            # pool.apply_async(DataDownloader.output_img, args=(out_dir, p_idx, maker, label_str, url))
            DataDownloader.output_img(out_dir, p_idx, maker, label_str, url)  # 单进程调试
            if (idx+1) % 1000 == 0:
                print('[Info] idx: {}'.format(idx+1))
                # break  # 测试

        # 多进程逻辑
        pool.close()
        pool.join()

        print('[Info] 图像下载完成: {}'.format(data_path))

    def process_excel_2_urls(self, data_path):
        """
        处理excel文件
        """
        # 标签: ["index", "markTime", "checkLabel", "label", "checker", "url", "marker", "_id", "taskId"]
        # 不指定names, 指定header，则数据以header的文本为keys
        pd_data = pd.read_csv(
            data_path,
            header=0,
            sep=";"
            # names=["index", "markTime", "checkLabel", "label", "checker", "url", "marker", "_id", "taskId"]
        )

        labels = pd_data["label"]
        urls = pd_data["url"]
        indexes = pd_data["index"]
        makers = pd_data["marker"]

        out_str_list = []
        for idx, (p_idx, maker, label_str, url) in enumerate(zip(indexes, makers, labels, urls)):
            is_box, box_list = DataDownloader.get_boxes(label_str)  # 是否写入box
            if not is_box:
                continue
            out_name = "{}_{}_{}.jpg".format(idx, maker, len(box_list))
            out_str = url + "," + out_name
            out_str_list.append(out_str)

        print('[Info] {} 处理完成, 样本数: {}'.format(data_path, len(out_str_list)))
        return out_str_list

    def process_data_of_wc(self, data_path):
        """
        处理语文和英语数据，来源于王超
        :param data_path:
        :return:
        """
        data_lines = read_file(data_path)
        file_name = os.path.basename(data_path)
        file_name_x = file_name.split('.')[0]

        out_str_list = []
        for idx, data_line in enumerate(data_lines):
            data_line = data_line.replace("'", "\"")
            data_dict = json.loads(data_line)
            url_str = data_dict['url']
            img_name = "{}_{}.jpg".format(file_name_x, idx)
            out_str = url_str + "," + img_name
            out_str_list.append(out_str)

        print('[Info] {} 处理完成, 样本数: {}'.format(data_path, len(out_str_list)))
        return out_str_list

    def process_folder_2_imgs(self, task_folder, out_dir):
        """
        处理文件夹
        """
        mkdir_if_not_exist(out_dir)
        paths_list, names_list = traverse_dir_files(task_folder)

        for path, name in zip(paths_list, names_list):
            self.process_excel(path, out_dir)  # 处理excel文件

        print('[Info] 全部下载完成')

    def process_folder_2_urls(self, task_folder, out_format, mode=0):
        """
        将样本数据写入URL+文件名的形式
        :param task_folder: 待处理样本文件的文件夹
        :param out_format: 输出文本格式
        :param mode: 0是正盛, 1是王超, 2是佳宝
        :return: None
        """
        paths_list, names_list = traverse_dir_files(task_folder)

        out_list = []
        for path, name in zip(paths_list, names_list):
            if mode == 0:
                out_str_list = self.process_excel_2_urls(path)  # 处理excel文件
            elif mode == 1:
                out_str_list = self.process_data_of_wc(path)
            else:
                continue
            out_list += out_str_list

        print('[Info] 总样本数: {}'.format(len(out_list)))

        out_file = out_format.format(len(out_list))
        write_list_to_file(out_file, out_list)
        print('[Info] 全部处理完成: {}'.format(out_file))


def save_imgs():
    task_folder = os.path.join(DATA_DIR, 'task_files')
    out_dir = os.path.join(DATA_DIR, 'task-raw-out')
    dd = DataDownloader()
    dd.process_folder_2_imgs(task_folder, out_dir)


def save_urls():
    task_folder = os.path.join(DATA_DIR, 'task_files_jiabao')  # mode 0
    # task_folder = os.path.join(DATA_DIR, 'task_files_yuwen_yingyu_wc')  # mode 1
    out_format = os.path.join(DATA_DIR, 'task-raw-out-{}.jb.txt')
    dd = DataDownloader()
    dd.process_folder_2_urls(task_folder, out_format, mode=0)


def merge_files():
    file_folder = os.path.join(DATA_DIR, 'task_urls_files')
    file_format = os.path.join(DATA_DIR, 'task_urls.{}.txt')
    paths_list, names_list = traverse_dir_files(file_folder)

    count = 0
    out_str_list = []
    for path, name in zip(paths_list, names_list):
        data_lines = read_file(path)
        for data_line in data_lines:
            url, _ = data_line.split(',')
            img_name = str(count).zfill(6) + ".jpg"
            out_str_list.append(url + "," + img_name)
            count += 1
    file_path = file_format.format(len(out_str_list))
    write_list_to_file(file_path, out_str_list)


def main():
    merge_files()


if __name__ == '__main__':
    main()