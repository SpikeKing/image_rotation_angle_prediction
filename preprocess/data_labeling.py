#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 20.11.20
"""

import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myutils.project_utils import *
from myutils.cv_utils import *
from x_utils.oss_utils import save_img_2_oss
from x_utils.vpf_utils import get_uc_rotation_vpf_service
from multiprocessing.pool import Pool


class DataLabeling(object):
    def __init__(self):
        pass

    @staticmethod
    def process_res_dict(res_dict, img_oss_name):
        data_dict = res_dict['data']
        angle = float(data_dict['prob'])
        img_url = data_dict['image_url']
        is_ok, parsed_image = download_url_img(img_url)
        if is_ok:
            img_r = rotate_img_with_bound(parsed_image, -angle)
            img_r = resize_img_fixed(img_r, 512, is_height=False)
            img_oss_url = save_img_2_oss(img_r, img_oss_name,
                                         "zhengsheng.wcl/datasets/problems-segment/pre-labeled-20201120")
            # show_img_bgr(img_r)
        else:
            img_oss_url = ""
        # print('[Info] 上传URL: {}'.format(img_oss_url))
        return img_oss_url

    @staticmethod
    def process_file(path, name, out_dir):
        data_lines = read_file(path)
        out_file = os.path.join(out_dir, '{}_out.txt'.format(name))
        for idx, data in enumerate(data_lines):
            try:
                img_id, img_url, _ = data.split('<sep>')
                data_dict = get_uc_rotation_vpf_service(img_url)
                is_success = data_dict['success']
                img_oss_name = "{}_{}.jpg".format(name, idx)
                if is_success:
                    img_oss_url = DataLabeling.process_res_dict(data_dict, img_oss_name)
                    if img_oss_url:
                        write_line(out_file, img_oss_url)
            except Exception as e:
                print('[Exception] {}'.format(e))
                continue
            # if idx == 10:
            #     break
        print('[Info] 处理完成! {}'.format(path))

    def process_old(self):
        img_dir = os.path.join(DATA_DIR, '2020_11_20')
        out_dir = os.path.join(DATA_DIR, '2020_11_20_out')
        mkdir_if_not_exist(out_dir)

        pool = Pool(processes=80)
        paths_list, names_list = traverse_dir_files(img_dir)
        for path, name in zip(paths_list, names_list):
            # DataLabeling.process_file(path, name, out_dir)
            pool.apply_async(DataLabeling.process_file, (path, name, out_dir))

        pool.close()
        pool.join()

    @staticmethod
    def process_labeled(idx, data_line, out_dir):
        oss_root_dir = "zhengsheng.wcl/problems_rotation/datasets/prelabeled_diff_20201130"
        url, angle = data_line.split(',')
        is_ok, img_bgr = download_url_img(url)
        if not is_ok:
            return

        img_bgr = rotate_img_for_4angle(img_bgr, int(angle))
        h, w, _ = img_bgr.shape
        ratio = safe_div(h, w)
        if ratio >= 1.0:
            url_1 = save_img_2_oss(img_bgr, "{}_v.jpg".format(idx), oss_root_dir)
            img_bgr = rotate_img_for_4angle(img_bgr, 270)
            url_2 = save_img_2_oss(img_bgr, "{}_vx.jpg".format(idx), oss_root_dir)
        else:
            url_1 = save_img_2_oss(img_bgr, "{}_h.jpg".format(idx), oss_root_dir)
            img_bgr = rotate_img_for_4angle(img_bgr, 180)
            url_2 = save_img_2_oss(img_bgr, "{}_hx.jpg".format(idx), oss_root_dir)

        out_path = os.path.join(out_dir, 'data_{}.txt'.format(angle))
        write_line(out_path, url_1)
        write_line(out_path, url_2)
        print('[Info] idx: {}'.format(idx))

    def process(self):
        in_path = os.path.join(DATA_DIR, '2020_11_26_same.txt')
        out_dir = os.path.join(DATA_DIR, '2020_11_26_same_labeled')
        mkdir_if_not_exist(out_dir)
        data_lines = read_file(in_path)
        pool = Pool(processes=30)

        for idx, data_line in enumerate(data_lines):
            # DataLabeling.process_labeled(idx, data_line, out_dir)
            pool.apply_async(DataLabeling.process_labeled, (idx, data_line, out_dir))

        pool.close()
        pool.join()

        print('[Info] 处理完成: {}'.format(out_dir))


def main():
    dl = DataLabeling()
    dl.process()


if __name__ == '__main__':
    main()
