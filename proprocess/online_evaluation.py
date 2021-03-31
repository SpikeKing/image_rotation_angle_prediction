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
from myutils.cv_utils import *
from x_utils.vpf_utils import get_trt_rotation_vpf_service, get_dmy_rotation_vpf_service, get_uc_rotation_vpf_service, \
    get_rotation_service_v3, get_rotation_service_v4, get_ocr_angle_service_v4, get_rotation_service_v4_1


class OnlineEvaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def process_url(img_url, mode="trt"):
        angle = -1
        out_url = ""
        if mode == "v3":
            res_dict = get_rotation_service_v3(img_url)
            angle = int(res_dict['data']['angle'])
        elif mode == "v4":
            res_dict = get_rotation_service_v4(img_url)
            angle = int(res_dict['data']['angle'])
            out_url = res_dict['data']['rotated_image_url']
        elif mode == "v4.1":
            res_dict = get_rotation_service_v4_1(img_url)
            angle = int(res_dict['data']['angle'])
            out_url = res_dict['data']['rotated_image_url']

        return angle, out_url

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

    @staticmethod
    def process_url_and_download(data_line, out_dir, idx, names_list):
        url = data_line
        name = url.split("?")[0].split("/")[-1]
        if name in names_list:
            print('[Info] 已处理: {}'.format(name))
            return
        is_ok, img_bgr = download_url_img(url)
        angel_new = OnlineEvaluation.process_url(url, mode="v4")
        img_bgr = rotate_img_for_4angle(img_bgr, angel_new)
        out_file = os.path.join(out_dir, name)
        print("[Info] idx: {}, out_file: {}".format(idx, out_file))
        cv2.imwrite(out_file, img_bgr)

    def evaluate_url_and_download(self):
        """
        处理数据
        """
        in_file = os.path.join(DATA_DIR, 'pingce_angle_20210302_2000.txt')
        out_dir = os.path.join(DATA_DIR, 'test_2000_20210302')

        paths_list, names_list = traverse_dir_files(out_dir)
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(in_file)

        pool = Pool(processes=40)

        for idx, data_line in enumerate(data_lines):
            OnlineEvaluation.process_url_and_download(data_line, out_dir, idx, names_list)
            # pool.apply_async(OnlineEvaluation.process_url_and_download, (data_line, out_dir, idx, names_list))

        pool.close()
        pool.join()

        print('[Info] 处理完成')


    @staticmethod
    def process_thread_right(idx, url, r_angle, out_file, write_dir=None):
        """
        多进程处理，有正确角度
        """
        # print('[Info] idx: {}, url: {}'.format(idx, url))
        r_angle = int(r_angle)
        angel_old, _ = OnlineEvaluation.process_url(url, mode="v4")
        angel_old = int(angel_old)
        angel_new, out_v4_url = OnlineEvaluation.process_url(url, mode="v4.1")
        angel_new = int(angel_new)
        print('[Info] idx: {},  r_angle: {}, angel_v3: {}, angel_v4: {}'
              .format(idx, r_angle, angel_old, angel_new))
        is_old = 0 if r_angle == angel_old else 1
        is_new = 0 if r_angle == angel_new else 1
        is_diff = 0 if angel_old == angel_new else 1
        out_line = ",".join([url, str(r_angle), str(angel_old), str(angel_new),
                             str(is_old), str(is_new), str(is_diff)])
        write_line(out_file, out_line)
        if write_dir:
            if out_v4_url:
                is_ok, img_bgr = download_url_img(out_v4_url)
                img_name = url.split('?')[0].split('/')[-1]
                write_path = os.path.join(write_dir, img_name)
                cv2.imwrite(write_path, img_bgr)  # 写入图像
            else:
                is_ok, img_bgr = download_url_img(url)
                img_rotated = rotate_img_for_4angle(img_bgr, angel_new)
                img_name = url.split('?')[0].split('/')[-1]
                write_path = os.path.join(write_dir, img_name)
                cv2.imwrite(write_path, img_rotated)  # 写入图像

    @staticmethod
    def process_save_img_url(idx, url, r_angle, out_file, write_dir=None):
        angel_new, out_v4_url = OnlineEvaluation.process_url(url, mode="v4.1")
        angel_new = int(angel_new)
        if angel_new == 0:
            print('[Info] idx: {},  r_angle: {}, angel_v4: {}'.format(idx, r_angle, angel_new))
            out_line = out_v4_url
            write_line(out_file, out_line)  # 写入行

    def evaluate_csv_right(self):
        """
        评估CSV文件
        """
        # in_file_name = 'test_400_right'     # 测试400
        # in_file_name = 'test_1000_right'    # 测试1000
        # in_file_name = 'random_1w_urls'    # 测试1w
        # in_file = os.path.join(DATA_DIR, 'test_urls_files', in_file_name + ".csv")

        # in_file_name = "sanghu.zj_question_cut_sampled_jueying_url_5k_1229"
        # in_file_name = "dump纯手写图片公式文本标注.out"
        # in_file_name = "7_train_原始图像.out"
        # in_file_name = "HW_TRAIN.out"
        # in_file_name = "biaozhu_fix.check"
        # in_file_name = "biaozhu_csv_out"
        # in_file_name = "angle_ds_answer_20210323.filter"
        # in_file_name = "angle_ds_question_20210323.filter"
        in_file_name = "angle_ds_solution_20210323.filter"
        in_file = os.path.join(DATA_DIR, 'page_dataset_files', in_file_name+".txt")  # 输入文件

        print('[Info] in_file: {}'.format(in_file))

        data_lines = read_file(in_file)
        print('[Info] 样本总量: {}'.format(len(data_lines)))
        if len(data_lines) == 0:
            print('[Info] 文件路径错误: {}'.format(in_file))
            return

        # 测试文件
        # if len(data_lines) > 2000:
        #     random.seed(47)
        #     # random.seed(89)
        #     random.shuffle(data_lines)  # 随机生成
        #     data_lines = data_lines[:2000]

        print('[Info] 样本数量: {}'.format(len(data_lines)))

        # 测试文件
        # time_str = get_current_time_str()
        # out_name = 'check_{}.{}.csv'.format(in_file_name, time_str)
        # out_dir = os.path.join(DATA_DIR, "check_dir_20210329")
        # mkdir_if_not_exist(out_dir)
        # out_file = os.path.join(out_dir, out_name)

        # 筛选文件
        out_dir = os.path.join(DATA_DIR, "xiaotu_dir")
        in_file_name = '{}_good.txt'.format(in_file_name)
        mkdir_if_not_exist(out_dir)
        out_file = os.path.join(out_dir, in_file_name)

        # write_dir = os.path.join(out_dir, 'write_dir_{}'.format(time_str))
        # mkdir_if_not_exist(write_dir)
        write_dir = None

        pool = Pool(processes=100)
        for idx, data_line in enumerate(data_lines):
            # 方案1
            # if idx == 0:
            #     continue
            # url, r_angle = data_line.split(',')

            # 方案2
            url, r_angle = data_line, 0

            name = url.split('/')[-1].split('.')[0]
            file_name_x = in_file_name.split('.')[0]
            url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_rotation/" \
                  "datasets/{}_x/{}.jpg".format(file_name_x, name)

            try:
                # pool.apply_async(OnlineEvaluation.process_thread_right, (idx, url, r_angle, out_file, write_dir))
                # OnlineEvaluation.process_thread_right(idx, url, r_angle, out_file, write_dir)

                # 筛选图像
                pool.apply_async(OnlineEvaluation.process_save_img_url, (idx, url, r_angle, out_file, write_dir))
                # OnlineEvaluation.process_save_img_url(idx, url, r_angle, out_file, write_dir)
            except Exception as e:
                continue

        pool.close()
        pool.join()

        print('[Info] 写入文件: {}'.format(out_file))


def main():
    oe = OnlineEvaluation()
    oe.evaluate_csv_right()
    # oe.evaluate_url_and_download()


if __name__ == '__main__':
    main()
