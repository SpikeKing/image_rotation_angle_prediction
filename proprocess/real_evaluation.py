#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 25.11.20
"""

import os
import cv2

from myutils.cv_utils import format_angle
from myutils.project_utils import *
from x_utils.vpf_utils import *
from root_dir import DATA_DIR


class RealEvaluation(object):
    def __init__(self):
        pass

    def generate_real_image(self):
        file_path = os.path.join(DATA_DIR, 'test_1000.txt')
        out_dir = os.path.join(DATA_DIR, 'test_1000_out')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(file_path)
        for idx, data_line in enumerate(data_lines):
            img_url, angle = data_line.split('\t')
            print('[Info] img_url: {}, angle: {}'.format(img_url, angle))
            is_ok, img_bgr = download_url_img(img_url)
            angle = int(angle)
            print('[Info] angle: {}'.format(angle))
            if angle == 90:
                img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
            elif angle == 270:
                img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                img_rotated = img_bgr

            out_path = os.path.join(out_dir, '{}.jpg'.format(idx))
            cv2.imwrite(out_path, img_rotated)
            print('[Info] 完成: {}'.format(out_path))
            # if idx == 10:
            #     break

    def process(self):
        file_path = os.path.join(DATA_DIR, 'test_1000.txt')
        out_file = os.path.join(DATA_DIR, 'test_1000_res.{}.txt'.format(get_current_time_str()))

        data_lines = read_file(file_path)

        for idx, data_line in enumerate(data_lines):
            img_url, angle = data_line.split('\t')
            img_url = img_url.split('?')[0]
            print('[Info] img_url: {}, angle: {}'.format(img_url, angle))
            angle = int(angle)
            try:
                uc_dict = get_uc_rotation_vpf_service(img_url)
                uc_angle = int(uc_dict['data']['angle'])
            except Exception as e:
                uc_angle = -1
            print('[Info] uc_angle: {}'.format(uc_angle))
            try:
                dmy_dict = get_dmy_rotation_vpf_service(img_url)
                dmy_angle = int(dmy_dict['data']['angel'])
                dmy_angle = format_angle(dmy_angle)
                dmy_angle = (360 - dmy_angle) % 360
            except Exception as e:
                dmy_angle = -1
            print('[Info] dmy_angle: {}'.format(dmy_angle))

            is_dmy = 1 if dmy_angle == angle else 0
            is_uc = 1 if uc_angle == angle else 0

            out_items = [img_url, str(angle), str(dmy_angle), str(is_dmy), str(uc_angle), str(is_uc)]
            print('[Info] out_items: {}'.format(out_items))
            out_line = ",".join(out_items)
            write_line(out_file, out_line)
            print('[Info] idx: {}'.format(idx))
            print('-' * 50)

    def generate_angle_imgs(self):
        img_dir = os.path.join(DATA_DIR, 'test_1000_out')
        out_dir = os.path.join(DATA_DIR, 'test_1000_out_4a')
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(img_dir)
        for path, name in zip(paths_list, names_list):
            name = name.split('.')[0]
            img_bgr = cv2.imread(path)
            for angle in [0, 90, 180, 270]:
                out_path = os.path.join(out_dir, '{}_{}.jpg'.format(name, angle))
                if angle == 90:
                    img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
                elif angle == 270:
                    img_rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    img_rotated = img_bgr
                cv2.imwrite(out_path, img_rotated)
            print('[Info] path: {}'.format(path))

        print('[Info] 全部处理完成: {}'.format(out_dir))


def main():
    reo = RealEvaluation()
    # reo.process()
    reo.generate_angle_imgs()


if __name__ == '__main__':
    main()