#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 6.1.21
"""

import collections
import json
import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from root_dir import DATA_DIR


class DatasetGeneratorV2(object):
    """
    数据集生成
    """
    def __init__(self):
        pass

    @staticmethod
    def convert(iw, ih, box):
        """
        将标注的xml文件标注转换为darknet形的坐标
        """
        iw = float(iw)
        ih = float(ih)
        dw = 1. / iw
        dh = 1. / ih
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def generate_file(file_path, file_idx, out_file_path):
        file_idx = str(file_idx).zfill(4)
        print('[Info] file_path: {}, file_idx: {}'.format(file_path, file_idx))

        url_format = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/yjb219735/ori_imgs/{}.jpg"

        print('[Info] 处理数据开始: {}'.format(file_path))
        data_line = read_file(file_path)[0]
        data_dict = json.loads(data_line)
        print('[Info] keys: {}'.format(data_dict.keys()))
        images = data_dict['images']

        id_name_dict = {}
        for idx, img in enumerate(images):
            img_id = img['id']
            image_name = img['file_name'].split('.')[0]
            height = img['height']
            width = img['width']

            # print('[Info] img: {}'.format(img))
            # print('[Info] img_id: {}, file_name: {}'.format(img_id, image_name))
            id_name_dict[img_id] = [image_name, height, width]
            # if idx == 20:
            #     break

        annotations = data_dict["annotations"]

        image_dict = collections.defaultdict(list)
        for idx, anno in enumerate(annotations):
            image_id = anno['image_id']
            image_name, ih, iw = id_name_dict[image_id]
            wh_box = anno['bbox']
            bbox = [wh_box[0], wh_box[1], wh_box[0] + wh_box[2], wh_box[1] + wh_box[3]]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            bbox_yolo = DatasetGeneratorV2.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            # print('[Info] image_id: {}, ih: {}, iw: {}, bbox: {}, bbox_yolo: {}'
            #       .format(image_name, ih, iw, bbox, bbox_yolo))

            image_dict[image_name].append(" ".join(["0", *bbox_yolo]))

        print('[Info] 样本数: {}'.format(len(image_dict.keys())))

        image_name_list = list(image_dict.keys())

        img_urls, error_urls = [], []
        n_error = 0
        for idx, image_name in enumerate(image_name_list):
            # print('[Info] idx: {}'.format(idx))
            bbox_yolo_list = image_dict[image_name]
            image_url = url_format.format(image_name)
            if len(bbox_yolo_list) == 0:
                n_error += 1
                error_urls.append(image_url)
                continue
            img_urls.append(image_url)

        print('[Info] 全部URL: {}, 正确: {}, 错误: {}'.format(len(image_name_list), len(img_urls), len(error_urls)))

        # write_list_to_file(out_file_path, img_urls)
        # write_list_to_file(err_file_path, error_urls)

        print('[Info] 处理完成! {}'.format(file_path))
        return img_urls

    @staticmethod
    def generate_file_v2(file_path, out_path):
        data_lines = read_file(file_path)
        print('[Info] 文件行数: {}'.format(len(data_lines)))
        url_list = []
        for data_line in data_lines:
            data_line = data_line.replace("\'", "\"")
            data_dict = json.loads(data_line)
            url = data_dict['url']
            print('[Info] url: {}'.format(url))
            url_list.append(url)
        write_list_to_file(out_path, url_list)
        print('[Info] 结果: {}'.format(out_path))

def process():
    dir_path = os.path.join(DATA_DIR, 'write_dataset_raw')
    paths_list, names_list = traverse_dir_files(dir_path)

    out_dataset_dir = os.path.join(DATA_DIR, 'write_dataset_txt')
    mkdir_if_not_exist(out_dataset_dir)
    out_file_path = os.path.join(out_dataset_dir, 'write_img_urls.{}.txt'.format(get_current_time_str()))
    # err_file_path = os.path.join(out_dataset_dir, 'err_urls.{}.txt'.format(get_current_time_str()))

    pool = Pool(processes=20)

    all_img_urls = []
    for file_idx, (path, name) in enumerate(zip(paths_list, names_list)):
        img_urls = DatasetGeneratorV2.generate_file(path, file_idx, out_file_path)
        all_img_urls += img_urls
        # print('[Info] path: {}'.format(path))
        # pool.apply_async(DatasetGeneratorV2.generate_file, (path, file_idx))

    print('[Info] 数据: {}'.format(len(all_img_urls)))
    all_img_urls = list(set(all_img_urls))
    print('[Info] 数据: {}'.format(len(all_img_urls)))
    all_img_urls = sorted(all_img_urls)
    write_list_to_file(out_file_path, all_img_urls)

    pool.close()
    pool.join()
    print('[Info] 全部处理完成: {}'.format(dir_path))


def process_v2():
    file_path = os.path.join(DATA_DIR, 'write_dataset_raw', 'dump纯手写图片公式文本标注.txt')
    out_path = os.path.join(DATA_DIR, 'write_dataset_txt', 'dump纯手写图片公式文本标注.out.txt')
    DatasetGeneratorV2.generate_file_v2(file_path, out_path)


def main():
    process_v2()


if __name__ == '__main__':
    main()
