#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 2.12.20
"""
import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATA_DIR
from myutils.project_utils import *
from x_utils.vpf_utils import get_trt_rotation_vpf_service, get_dmy_rotation_vpf_service, get_uc_rotation_vpf_service


class OnlineEvaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def process_url(img_url, mode="trt"):
        angle = -1
        if mode == "trt":
            res_dict = get_trt_rotation_vpf_service(img_url)
            angle = res_dict['data']['angle']
        if mode == "dmy":
            res_dict = get_dmy_rotation_vpf_service(img_url)
            angle = res_dict['data']['angel']
        if mode == "test":
            res_dict = get_uc_rotation_vpf_service(img_url)
            angle = res_dict['data']['angle']
        return angle

    def init_urls(self):
        urls_path = os.path.join(DATA_DIR, 'long_text_2020-12-02-09-44-42.txt')
        out_path = os.path.join(DATA_DIR, 'test_400_res.right.txt')
        urls = read_file(urls_path)
        for idx, url in enumerate(urls):
            url = url.split("?")[0]
            uc_angle = self.process_url(img_url=url, mode="trt")
            dmy_angle = self.process_url(img_url=url, mode="dmy")
            # url,r_angle,dmy_angle,is_dmy,uc_angle,is_uc
            r_angle = uc_angle  # 以我们的角度为基准
            is_dmy = 1 if r_angle == dmy_angle else 0
            is_uc = 1 if r_angle == uc_angle else 0
            out_items = [url, str(r_angle), str(dmy_angle), str(is_dmy), str(uc_angle), str(is_uc)]
            print('[Info] {} out_items: {}'.format(idx, out_items))
            out_line = ",".join(out_items)
            write_line(out_path, out_line)
        print('[Info] 处理完成: {}'.format(out_path))

    @staticmethod
    def update_one_url(idx, data_line, out_path):
        items = data_line.split(',')
        url = items[0]
        r_angle = items[1]
        dmy_angle = items[2]
        uc_angle = OnlineEvaluation.process_url(img_url=url, mode="trt")

        # url,r_angle,dmy_angle,is_dmy,uc_angle,is_uc
        is_dmy = 1 if r_angle == dmy_angle else 0
        is_uc = 1 if r_angle == uc_angle else 0
        out_items = [url, str(r_angle), str(dmy_angle), str(is_dmy), str(uc_angle), str(is_uc)]
        print('[Info] {} out_items: {}'.format(idx, out_items))
        out_line = ",".join(out_items)
        write_line(out_path, out_line)

    def update_urls(self):
        urls_path = os.path.join(DATA_DIR, 'test_1000_res.right.csv')
        out_path = os.path.join(DATA_DIR, 'test_1000_res.right.{}.csv'.format(get_current_time_str()))
        data_lines = read_file(urls_path)

        pool = Pool(processes=30)
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            # OnlineEvaluation.update_one_url(idx, data_line, out_path)  # 更新URL
            pool.apply_async(OnlineEvaluation.update_one_url, (idx, data_line, out_path))
        print('[Info] 处理完成: {}'.format(out_path))
        pool.close()
        pool.join()

    def evaluate_1000_right(self):
        """
        处理数据v3
        """
        in_file = os.path.join(DATA_DIR, 'test_1000_res.right.e2.csv')
        data_lines = read_file(in_file)
        out_list = []

        n_old_right, n_right, n_all, n_error = 0, 0, 0, 0
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            url, r_angle, dmy_angle, is_dmy, uc_angle, is_uc = data_line.split(',')

            uc_angle = int(uc_angle)
            uc_is_ok = int(is_uc)
            r_angle = int(r_angle)

            x_angle = self.process_url(url, mode="test")
            x_angle = int(x_angle)

            x_is_ok = 1 if x_angle == r_angle else 0
            if uc_is_ok == 1:
                n_old_right += 1
            if x_angle == r_angle:
                print('[Info] {} 预测正确 {} - {}! {}'.format(idx, x_angle, r_angle, url))
                n_right += 1
            else:
                print('[Info] {} 预测错误 {} - {}! {}'.format(idx, x_angle, r_angle, url))
                n_error += 1
            n_all += 1

            out_list.append([url, r_angle, dmy_angle, is_dmy, uc_angle, uc_is_ok, x_angle, x_is_ok])

        print('[Info] 最好正确率: {} - {} / {}'.format(safe_div(n_old_right, n_all), n_old_right, n_all))
        print('[Info] 当前正确率: {} - {} / {}'.format(safe_div(n_right, n_all), n_right, n_all))

        out_file = os.path.join(DATA_DIR, 'check_{}.e{}.xlsx'.format(safe_div(n_right, n_all), n_error))
        write_list_to_excel(
            out_file,
            ["url", "r_angle", "dmy_angle", "is_dmy", "uc_angle", "uc_is_ok", "x_angle", "x_is_ok"],
            out_list
        )


def main():
    oe = OnlineEvaluation()
    oe.evaluate_1000_right()


if __name__ == '__main__':
    main()
