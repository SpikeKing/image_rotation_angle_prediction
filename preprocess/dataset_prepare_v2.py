#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 10.12.20
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from x_utils.vpf_utils import *
from root_dir import DATA_DIR
from multiprocessing.pool import Pool


class DatasetPrepareV2(object):
    def __init__(self):
        self.data_dir = os.path.join(DATA_DIR, '2020_12_09')  # 数据文件夹
        self.out_dir = os.path.join(DATA_DIR, '2020_12_09_out')  # 输出文件夹
        mkdir_if_not_exist(self.out_dir)
        self.out_dir_v2 = os.path.join(DATA_DIR, '2020_12_09_out_v2_19791263')  # 输出文件夹
        mkdir_if_not_exist(self.out_dir_v2)
        self.out_dir_v3 = os.path.join(DATA_DIR, '2020_12_09_out_v3')  # 输出文件夹
        mkdir_if_not_exist(self.out_dir_v3)


    @staticmethod
    def get_wh_from_url(url_str):
        """
        判断图像URL的宽高
        URL示例:xxx.jpg?width=1626&amp;height=1024
        """
        try:
            x_width_list = re.findall(r".*width=(.+)&amp;", url_str)
            if len(x_width_list) == 0:
                x_width_list = re.findall(r".*width=(.+)", url_str)
            img_width = int(float(x_width_list[0]))

            x_height_list = re.findall(r".*height=(.+)&amp;", url_str)
            if len(x_height_list) == 0:
                x_height_list = re.findall(r".*height=(.+)", url_str)
            img_height = int(float(x_height_list[0]))
        except Exception as e:
            img_width = -1
            img_height = -1

        return img_width, img_height

    @staticmethod
    def process_one_file(out_dir, path, name):
        pre_iw, pre_ih = -1, -1
        out_path = os.path.join(out_dir, name + ".out.txt")
        print('[Info] 输出文件: {}'.format(out_path))
        create_file(out_path)
        out_url_list = []
        data_lines = read_file(path)
        for data_line in data_lines:
            items = data_line.split('<sep>')
            url = items[1]
            iw, ih = DatasetPrepareV2.get_wh_from_url(url)
            if pre_iw == iw and pre_ih == ih:
                continue
            url = url.split('?')[0]
            out_url_list.append(url)
            pre_iw, pre_ih = iw, ih
        write_list_to_file(out_path, out_url_list)
        print('[Info] 处理完成: {}, {}'.format(out_path, len(out_url_list)))

    def process_urls(self):
        paths_list, names_list = traverse_dir_files(self.data_dir)
        print('[Info] 文件数: {}'.format(len(paths_list)))

        pool = Pool(processes=40)

        for path, name in zip(paths_list, names_list):
            # DatasetPrepareV2.process_one_file(self.out_dir, path, name)
            pool.apply_async(DatasetPrepareV2.process_one_file, (self.out_dir, path, name))
            # break

        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(self.out_dir))

    def merge_urls_2_others(self):
        paths, names = traverse_dir_files(self.out_dir)

        out_lines = []
        for path, name in zip(paths, names):
            data_lines = read_file(path)
            out_lines += data_lines

        print('[Info] 图像数: {}'.format(len(out_lines)))

        out_lines = list(set(out_lines))
        print('[Info] 图像数: {}'.format(len(out_lines)))

        gap = 100000
        for i in range(0, len(out_lines), gap):
            s_idx = i
            e_idx = min(i + gap, len(out_lines))
            sub_out_lines = out_lines[s_idx: e_idx]
            out_path = os.path.join(self.out_dir_v2, "{}-{}.out.txt".format(s_idx, e_idx))
            create_file(out_path)
            write_list_to_file(out_path, sub_out_lines)
            print('[Info] 文件写入完成! {}'.format(out_path))

    @staticmethod
    def draw_polygon(img_bgr, box_list):
        import cv2
        triangle = np.array([box_list], np.int32)
        img_out = cv2.polylines(img_bgr, [triangle], True, (0, 255, 0), thickness=3)
        show_img_bgr(img_out)


    @staticmethod
    def parse_handwriting_and_angle(url, res_dict, is_show=False):
        """
        提取字符串中的手写信息
        """
        data_dict = res_dict['data']
        angle = (360 - int(data_dict['call_result']['angle'])) % 360
        # print('[Info] angle: {}'.format(angle))
        box_info_list = data_dict['box_info_list']

        if is_show:
            is_ok, img_bgr = download_url_img(url)
            img_bgr = rotate_img_with_bound(img_bgr, angle)

        hw_boxes = []  # 手写框
        for box_info in box_info_list:
            # print('[Info] box: {}'.format(box_info))
            rec_classify = box_info['recClassify']
            box = box_info['box']
            if rec_classify == 2:
                if is_show:
                    DatasetPrepareV2.draw_polygon(img_bgr, box)
                hw_boxes.append(box_info)

        if hw_boxes:
            out_dict = {
                "img_url": url,
                "angle": angle,  # 角度
                "hw_boxes": hw_boxes
            }
        else:
            out_dict = {}

        return out_dict

    @staticmethod
    def process_url(idx, url, out_path_list):
        """
        处理URL
        """
        print('[Info]' + '-' * 50)
        print('[Info] url: {}'.format(url))
        hw_path, angel_right_path, angel_error_path, processed_path = out_path_list

        res_dict = get_dmy_rotation_vpf_service(url)

        data_dict = res_dict['data']
        dmy_angle = int(data_dict['call_result']['angle'])
        dmy_angle = format_angle(dmy_angle)
        dmy_hw_dict = DatasetPrepareV2.parse_handwriting_and_angle(url, res_dict)
        print('[Info] dmy_angle: {}'.format(dmy_angle))

        uc_dict = get_uc_rotation_vpf_service(url)
        uc_angle = int(uc_dict['data']['angle'])
        print('[Info] uc_angle: {}'.format(uc_angle))

        if dmy_angle == uc_angle:
            write_line(angel_right_path, "{},{}".format(url, uc_angle))
            print('[Info] {} 角度相同! {}'.format(idx, uc_angle))
        else:
            write_line(angel_error_path, "{},{},{}".format(url, dmy_angle, uc_angle))
            print('[Info] {} 角度不同! {} - {}'.format(idx, dmy_angle, uc_angle))

        if dmy_hw_dict:
            write_line(hw_path, json.dumps(dmy_hw_dict))
            print('[Info] {} 包含手写: {}'.format(idx, dmy_hw_dict))

        write_line(processed_path, '{}'.format(str(idx)))
        print('[Info] 处理完成: {}, {}'.format(idx, url))

    @staticmethod
    def process_url_wrapper(idx, url, out_path_list):
        # print('[Info]' + '-' * 20)
        try:
            DatasetPrepareV2.process_url(idx, url, out_path_list)
        except Exception as e:
            print('[Info] Exception: {} - {}'.format(idx, e))

    def generate_data(self):
        paths, names = traverse_dir_files(self.out_dir_v2)
        print('[Info] 文件数: {}'.format(len(paths)))

        for path, name in zip(paths, names):
            pool = Pool(processes=1)
            print('[Info] path: {}'.format(path))
            name_x = name.split('.')[0]

            hw_path = os.path.join(self.out_dir_v3, name_x + ".hw.txt")
            angel_right_path = os.path.join(self.out_dir_v3, name_x + ".right.txt")
            angel_error_path = os.path.join(self.out_dir_v3, name_x + ".error.txt")
            processed_path = os.path.join(self.out_dir_v3, name_x + ".pd.txt")

            # 输出路径列表
            out_path_list = [hw_path, angel_right_path, angel_error_path, processed_path]

            idx_list = read_file(processed_path)
            idx_list = [int(idx) for idx in idx_list]

            data_lines = read_file(path)
            print('[Info] samples: {}'.format(len(data_lines)))
            for idx, data_line in enumerate(data_lines):
                if idx == 60:
                    break

                if idx in idx_list:
                    print('[Info] idx: {} 已经处理完成!'.format(idx))
                    continue
                DatasetPrepareV2.process_url_wrapper(idx, data_line, out_path_list)
                # pool.apply_async(DatasetPrepareV2.process_url, (idx, data_line, out_path_list))

            pool.close()
            pool.join()
            break

        print('[Info] 处理完成: {}'.format(self.out_dir_v3))


def main():
    dp2 = DatasetPrepareV2()
    # dp2.process()
    # dp2.merge_urls_2_others()
    dp2.generate_data()


if __name__ == '__main__':
    main()