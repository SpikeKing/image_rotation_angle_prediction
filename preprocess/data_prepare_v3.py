#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 24.11.20
"""

import os
import sys
import shutil
import json

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR, ROOT_DIR
from myutils.project_utils import mkdir_if_not_exist, read_file, write_list_to_file, create_file, traverse_dir_files


class DataPrepareV3(object):
    def __init__(self):
        self.file_dir = os.path.join(DATA_DIR, 'train_data_v3')
        self.folder1_dir = os.path.join(self.file_dir, 'formula_dec')
        self.folder2_dir = os.path.join(self.file_dir, 'wrote_formula_dec')
        self.folder3_dir = os.path.join(self.file_dir, 'text_dec', 'csv')
        self.folder4_dir = os.path.join(self.file_dir, 'text_dec')
        self.folder5_dir = os.path.join(self.file_dir, 'formula_detection')

        self.file1_path = os.path.join(self.folder4_dir, 'raw_ciyubuquan.txt')
        self.file2_path = os.path.join(self.folder4_dir, 'raw_shouxietouzi.txt')
        self.file3_path = os.path.join(self.folder4_dir, 'text_formula_biaozhu_1.txt')
        self.file4_path = os.path.join(self.folder4_dir, 'yuwen_text_detection_1105_zl_1w_clean.csv')
        self.file5_path = os.path.join(self.folder4_dir, 'wrote_touzi_all.txt')
        self.file6_path = os.path.join(self.folder4_dir, 'jingbiao.txt')

        self.out_dir = os.path.join(DATA_DIR, 'train_data_v3_out')
        mkdir_if_not_exist(self.out_dir)
        self.out1_dir = os.path.join(self.out_dir, 'formula_dec_out')
        mkdir_if_not_exist(self.out1_dir)
        self.out2_dir = os.path.join(self.out_dir, 'wrote_formula_dec_out')
        mkdir_if_not_exist(self.out2_dir)
        self.out3_up_dir = os.path.join(self.out_dir, 'text_dec_out')
        mkdir_if_not_exist(self.out3_up_dir)
        self.out3_dir = os.path.join(self.out_dir, 'text_dec_out', 'csv_out')
        mkdir_if_not_exist(self.out3_dir)
        self.out5_dir = os.path.join(self.out_dir, 'formula_detection_out')
        mkdir_if_not_exist(self.out5_dir)

        self.out1_path = os.path.join(self.out_dir, 'text_dec_out', 'raw_ciyubuquan.out.txt')
        self.out2_path = os.path.join(self.out_dir, 'text_dec_out', 'raw_shouxietouzi.out.txt')
        self.out3_path = os.path.join(self.out_dir, 'text_dec_out', 'text_formula_biaozhu_1.out.txt')
        self.out4_path = os.path.join(self.out_dir, 'text_dec_out', 'yuwen_text_detection_1105_zl_1w_clean.out.txt')
        self.out5_path = os.path.join(self.out_dir, 'text_dec_out', 'wrote_touzi_all.out.txt')
        self.out6_path = os.path.join(self.out_dir, 'text_dec_out', 'jingbiao.out.txt')

        # self.file1_name = os.path.join(self.folder1_dir, "biaozhu_0921_10013.csv")
        # self.file1_out_format = os.path.join(self.out1_dir, "biaozhu_0921_10013.out.{}.txt")

    def check_label_num(self, label_str):
        label_list = json.loads(label_str)
        labels = label_list[0]
        n_labels = len(labels)
        # print('[Info] num of labels: {}'.format(n_labels))
        if n_labels >= 1:
            return True
        else:
            return False

    def process_csv(self, file_path, out_path):
        """
        处理CSV文件
        :param file_path: csv输入文件
        :param out_path: 输出文件
        :return: None
        """
        data_lines = read_file(file_path)  # 读取数据

        url_list = []
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            # print('[Info] data_line: {}'.format(data_line))
            items = data_line.split(';')
            # print('[Info] items: {}'.format(len(items)))
            label = items[3]
            url = items[5].replace("\"", "")
            # print('[Info] label: {}'.format(label))
            # print('[Info] url: {}'.format(url))
            if self.check_label_num(label):
                url_list.append(url)

        print('[Info] 样本数量: {}'.format(len(url_list)))
        write_list_to_file(out_path, url_list)
        print('[Info] 写入数据完成: {}'.format(out_path))

    def process_folder(self, folder_dir, out_folder):
        """
        处理CSV文件夹
        :param folder_dir: 文件夹
        :param out_folder: 输出文件夹
        :return: None
        """
        print('[Info] 待处理文件夹: {}'.format(folder_dir))
        paths_list, names_list = traverse_dir_files(folder_dir, ext='csv')
        print('[Info] 文件数量: {}'.format(len(paths_list)))

        for path, name in zip(paths_list, names_list):
            print('[Info] path: {}'.format(path))
            file_name = name.split('.')[0]
            out_path = os.path.join(out_folder, '{}.out.txt'.format(file_name))
            create_file(out_path)
            self.process_csv(path, out_path)

    def process_txt(self, file_path, out_path):
        data_lines = read_file(file_path)  # 读取数据

        url_list = []
        for idx, data_line in enumerate(data_lines):
            print('[Info] data_line: {}'.format(data_line))
            items = data_line.split('\t')
            print('[Info] items: {}'.format(len(items)))
            url = items[0]
            url_list.append(url)

        write_list_to_file(out_path, url_list)
        print('[Info] 写入完成: {}'.format(out_path))

    def process_txt_v2(self, file_path, out_path):
        data_lines = read_file(file_path)  # 读取数据

        url_list = []
        for idx, data_line in enumerate(data_lines):
            # print('[Info] data_line: {}'.format(data_line))
            data_line = data_line.replace("\'", "\"")
            data_dict = json.loads(data_line)
            url = data_dict['url']
            url_list.append(url)

        write_list_to_file(out_path, url_list)
        print('[Info] 写入完成: {}'.format(out_path))

    def process_csv_v2(self, file_path, out_path):
        data_lines = read_file(file_path)  # 读取数据
        url_list = []
        for idx, data_line in enumerate(data_lines):
            # print('[Info] data_line: {}'.format(data_line))
            items = data_line.split(';')
            url = items[1]
            url_list.append(url)

        write_list_to_file(out_path, url_list)
        print('[Info] 写入完成: {}'.format(out_path))

    def process(self):
        # self.process_folder(self.folder1_dir, self.out1_dir)
        # self.process_folder(self.folder2_dir, self.out2_dir)
        # self.process_folder(self.folder3_dir, self.out3_dir)
        # self.process_txt(self.file1_path, self.out1_path)
        # self.process_txt(self.file2_path, self.out2_path)
        # self.process_txt_v2(self.file3_path, self.out3_path)
        # self.process_csv_v2(self.file4_path, self.out4_path)
        # self.process_txt(self.file5_path, self.out5_path)
        # self.process_txt(self.file6_path, self.out6_path)
        self.process_folder(self.folder5_dir, self.out5_dir)

    def merge(self):
        out_format = os.path.join(DATA_DIR, 'train_data_v3_out.{}.txt')
        paths_list, names_list = traverse_dir_files(self.out_dir)
        print('[Info] 总文本数: {}'.format(len(paths_list)))
        all_data_lines = []
        for path in paths_list:
            data_lines = read_file(path)
            all_data_lines += data_lines

        all_data_lines = sorted(list(set(all_data_lines)))
        out_path = out_format.format(len(all_data_lines))
        print('[Info] 总数据量: {}'.format(len(all_data_lines)))
        write_list_to_file(out_path, all_data_lines)
        print('[Info] 写入数据完成: {}'.format(out_path))

    def check(self):
        data_dir = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v3')
        paths_list, names_list = traverse_dir_files(data_dir, is_sorted=False)
        print('[Info] 文件数量: {}'.format(len(paths_list)))
        for path in paths_list:
            x_path = path.split("?")[0]
            shutil.move(path, x_path)
        print('[Info] 处理完成!')


def main():
    dpv = DataPrepareV3()
    # dpv.process()
    # dpv.merge()
    dpv.check()


if __name__ == '__main__':
    main()
