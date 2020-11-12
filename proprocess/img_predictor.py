#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 11.11.20
"""
import cv2
import copy
import json
import os
import sys
import numpy as np
import tensorflow.keras.backend as K

from tensorflow_core.python.keras.models import load_model

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import show_img_bgr, rotate_img_with_bound, merge_two_imgs, draw_text, resize_img_fixed
from myutils.project_utils import *

from root_dir import DATA_DIR
from utils import angle_error


class ImgPredictor(object):
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        """
        加载模型
        """
        model_location = os.path.join(DATA_DIR, 'models', 'problem_rotnet_resnet50_4.6894.hdf5')
        model = load_model(model_location, custom_objects={'angle_error': angle_error})
        return model

    def predict_img(self, test_img_bgr):
        """
        预测角度
        """
        # show_img_bgr(img_bgr)
        test_img_bgr = cv2.resize(test_img_bgr, (224, 224))
        test_img_bgr_b = np.expand_dims(test_img_bgr, axis=0)
        prediction = self.model.predict(test_img_bgr_b)
        angle = (360 - int(K.argmax(prediction[0]))) % 360
        # print('[Info] angle: {}'.format(angle))
        # out_img_bgr = rotate_img_with_bound(img_bgr, angle)
        # show_img_bgr(out_img_bgr)
        return angle

    def process_item(self, data_dict, out_dir):
        """
        处理单个数据
        """
        img_id = data_dict["id"]
        url = data_dict["url"]
        angle = data_dict["angle"]
        print('[Info] img_id: {}'.format(img_id))
        print('[Info] url: {}'.format(url))
        print('[Info] angle: {}'.format(angle))
        is_ok, img_bgr = download_url_img(url)
        s_time = time.time()
        p_angle = self.predict_img(img_bgr)
        elapsed_time = time.time() - s_time
        a = angle - p_angle
        abs_angle = abs((a + 180) % 360 - 180)

        if abs_angle > 5:
            img_bgr_d = rotate_img_with_bound(img_bgr, angle)
            img_bgr_p = rotate_img_with_bound(img_bgr, p_angle)
            img_merged = merge_two_imgs(img_bgr_d, img_bgr_p)
            out_file = os.path.join(out_dir, '{}_{}.jpg'.format(img_id, abs_angle))
            cv2.imwrite(out_file, img_merged)

        print('[Info] p_angle: {}'.format(p_angle))
        return [img_id, url, angle, p_angle, abs_angle, elapsed_time]

    def process(self):
        """
        处理全部数据
        """
        in_file = os.path.join(DATA_DIR, '2020_11_12_out.9979.txt')
        out_file_format = os.path.join(DATA_DIR, 'out_file.{}.xlsx')
        out_dir = os.path.join(DATA_DIR, 'out_imgs')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(in_file)
        print('[Info] 样本数: {}'.format(len(data_lines)))

        random.seed(47)
        random.shuffle(data_lines)  # shuffle

        res_list = []
        for idx, data_line in enumerate(data_lines):
            data_dict = json.loads(data_line)
            item_list = self.process_item(data_dict, out_dir)
            res_list.append(item_list)

            # img_bgr_d = rotate_img_with_bound(img_bgr, angle)
            # img_bgr_p = rotate_img_with_bound(img_bgr, p_angle)

            # img_bgr_d = resize_img_fixed(img_bgr_d, 512)
            # img_bgr_p = resize_img_fixed(img_bgr_p, 512)
            # draw_text(img_bgr_d, str(angle), org=(10, 50))
            # draw_text(img_bgr_p, str(p_angle), org=(10, 50))

            # show_img_bgr(img_merged)

            # show_img_bgr(img_bgr_d)
            # show_img_bgr(img_bgr_p)

            print('-' * 10)
            if idx == 200:
                break

        titles = ["img_id", "url", "angle", "p_angle", "abs_angle", "elapsed_time"]
        out_file = out_file_format.format(len(res_list))
        write_list_to_excel(out_file, titles, res_list)
        print('[Info] 写入文件完成! {}'.format(out_file))


def main():
    ip = ImgPredictor()
    ip.process()


if __name__ == '__main__':
    main()