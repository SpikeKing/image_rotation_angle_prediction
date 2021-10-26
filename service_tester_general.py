#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 31.8.21
"""

import argparse
from multiprocessing.pool import Pool

from myutils.cv_utils import *
from myutils.make_html_page import make_html_page
from myutils.project_utils import *
from x_utils.vpf_sevices import get_vpf_service_np


class ServiceTesterGeneral(object):
    """
    通用的服务测试逻辑
    """
    def __init__(self, in_folder, service, out_folder, max_num):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.service = service
        self.max_num = max_num
        print('[Info] 输入文件夹: {}'.format(self.in_folder))
        print('[Info] 服务: {}'.format(self.service))
        print('[Info] 输出文件夹: {}'.format(self.out_folder))
        print('[Info] 最大样本数: {}'.format(self.max_num))

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/Image-Rotation/imgs-tmp/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_img_path(img_idx, img_path, service,  out_file):
        img_bgr = cv2.imread(img_path)
        res_dict = get_vpf_service_np(img_np=img_bgr, service_name=service)  # 表格
        angle = res_dict["data"]["angle"]
        angle = int(angle)
        if angle != 0:
            img_name = img_path.split("/")[-1]
            img_url = ServiceTesterGeneral.save_img_path(img_bgr, img_name)
            write_line(out_file, "{}\t{}\t{}".format(img_url, angle, img_path))
        if img_idx % 1000 == 0:
            print('[Info] 处理完成: {}, angle: {}'.format(img_idx, angle))

    def process_folder(self):
        print('[Info] 测试文件夹: {}'.format(self.in_folder))
        file_paths, file_names = traverse_dir_files(self.in_folder)
        data_dict = dict()
        for file_path, file_name in zip(file_paths, file_names):
            data_lines = read_file(file_path)
            data_dict[file_name] = data_lines

        out_dict = dict()
        pool = Pool(processes=100)
        for file_name in data_dict.keys():
            data_lines = data_dict[file_name]
            random.seed(47)
            random.shuffle(data_lines)
            if len(data_lines) > self.max_num:
                data_lines = data_lines[:self.max_num]
            print('[Info] 文件名: {}, 样本数: {}'.format(file_name, len(data_lines)))

            time_str = get_current_time_str()
            type_name = file_name.split(".")[0]
            sub_folder = os.path.join(self.out_folder, type_name)
            out_file = os.path.join(sub_folder, "out_{}.txt".format(time_str))
            out_html = os.path.join(sub_folder, "out_{}.html".format(time_str))
            out_dict[file_name] = [out_file, out_html]
            for img_idx, img_path in enumerate(data_lines):
                pool.apply_async(ServiceTesterGeneral.process_img_path, (img_idx, img_path, self.service, out_file))
        pool.close()
        pool.join()
        print('[Info] 服务处理完成!')
        print('[Info] 最终结果: ')
        for file_name in out_dict.keys():
            out_file, out_html = out_dict[file_name]
            data_lines = data_dict[file_name]
            out_lines = read_file(out_file)
            right_rate = safe_div(len(data_lines) - len(out_lines), len(data_lines))
            print('[Info] \t文件: {}, 正确率: {}%'.format(file_name, right_rate * 100))
            out_list = []
            for data_line in data_lines:
                items = data_line.split("\t")
                out_list.append(items)
            make_html_page(out_html, out_list)
        print('[Info] 全部处理完成')


def parse_args():
    """
    处理脚本参数，支持相对路径
    """
    parser = argparse.ArgumentParser(description='服务测试')
    parser.add_argument('-i', dest='in_folder', required=False, help='测试文件夹', type=str)
    parser.add_argument('-s', dest='service', required=False, help='服务', type=str)
    parser.add_argument('-o', dest='out_folder', required=False, help='输出文件夹', type=str)
    parser.add_argument('-m', dest='max_num', required=False, help='最大测试量', type=int, default=-1)

    args = parser.parse_args()

    arg_in_folder = args.in_folder
    print("测试文件夹: {}".format(arg_in_folder))

    arg_service = args.service
    print("服务: {}".format(arg_service))

    arg_out_folder = args.out_folder
    print("输出文件夹: {}".format(arg_out_folder))
    mkdir_if_not_exist(arg_out_folder)

    arg_max_num = int(args.max_num)
    print("最大测试量: {}".format(arg_max_num))

    return arg_in_folder, arg_service, arg_out_folder, arg_max_num


def main():
    arg_in_folder, arg_service, arg_out_folder, arg_max_num = parse_args()
    st = ServiceTesterGeneral(arg_in_folder, arg_service, arg_out_folder, arg_max_num)
    st.process_folder()


if __name__ == '__main__':
    main()
