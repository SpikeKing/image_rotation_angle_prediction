#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 11.11.20
"""
import os
import sys
import collections

import cv2
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from x_utils.vpf_utils import *
from myutils.cv_utils import *
from myutils.project_utils import *

from root_dir import DATA_DIR


class ProblemTester(object):
    def __init__(self):
        # self.model_name = "rotnet_v3_mobilenetv2_base224_20201205_2.1.h5"
        # self.model_name = "rotnet_v3_mobilenetv2_448_20201206.2.hdf5"  # 效果不好
        # self.model_name = "rotnet_v3_resnet50_224_20201207.3.hdf5"
        self.model_name = "rotnet_v3_mobilenetv2_pg448_20201211.1.hdf5"
        print('[Info] model name: {}'.format(self.model_name))
        self.model = self.load_model()

    @staticmethod
    def save_pb_model(model):
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        # full_model = full_model.get_concrete_function(
        #     [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
        #      tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)])
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("[Info] Frozen model layers: ")
        for layer in layers:
            print(layer)
        print("-" * 60)
        print("[Info] Frozen model inputs: ")
        print(frozen_func.inputs)
        print("[Info] Frozen model outputs: ")
        print(frozen_func.outputs)

        # path of the directory where you want to save your model
        frozen_out_path = os.path.join(DATA_DIR, 'pb_models')
        # name of the .pb file
        frozen_graph_filename = "frozen_graph_{}".format(get_current_time_str())

        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=frozen_out_path,
                          name=f"{frozen_graph_filename}.pb",
                          as_text=False)
        # Save its text representation
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=frozen_out_path,
                          name=f"{frozen_graph_filename}.pbtxt",
                          as_text=True)
        print('[Info] 存储PB模型完成!')

    def load_model(self):
        """
        加载模型
        """
        model_location = os.path.join(DATA_DIR, 'models', self.model_name)
        model = tf.keras.models.load_model(model_location, compile=False)
        # self.save_pb_model(model)  # 存储pb模型
        return model

    @staticmethod
    def format_angle(angle):
        """
        格式化角度
        """
        if angle <= 45 or angle >= 325:
            r_angle = 0
        elif 45 < angle <= 135:
            r_angle = 90
        elif 135 < angle <= 225:
            r_angle = 180
        else:
            r_angle = 270
        return r_angle

    def predict_img_bgr_prob(self, img_bgr, img_size=448):
        img_list = []
        # img_size = (224, 224)
        img_size = (img_size, img_size)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, img_size)  # resize

        img_list.append(img_rgb)
        for idx in [90, 180, 270]:
            img_tmp = rotate_img_for_4angle(img_rgb, idx)
            img_list.append(img_tmp)

        imgs_arr = np.array(img_list)
        print('[Info] imgs_arr: {}'.format(imgs_arr.shape))

        imgs_arr_b = preprocess_input(imgs_arr)

        predictions = self.model.predict(imgs_arr_b)
        angle_dict = collections.defaultdict(int)
        for i in range(4):
            probs = predictions[i]
            angle = (int(K.argmax(probs))) * 90 % 360
            angle = (angle + 90 * i) % 360
            angle_dict[angle] += 1

        angle_list = sort_dict_by_value(angle_dict)
        angle = angle_list[0][0]
        return angle

    def predict_img_bgr(self, img_bgr):
        """
        预测角度
        """
        angle = self.predict_img_bgr_prob(img_bgr)
        # angle = self.format_angle(angle)
        return angle

    def predict_img_path(self, img_path):
        """
        预测图像路径
        """
        print('[Info] img_path: {}'.format(img_path))
        img_bgr = cv2.imread(img_path)
        angle = self.predict_img_bgr(img_bgr)
        print('[Info] 预测角度: {}'.format(angle))
        return angle

    def process_img_url(self, url):
        is_ok, img_bgr = download_url_img(url)
        angle = self.predict_img_bgr(img_bgr)
        return angle

    def process_img_vpf(self, img_url):
        # res_dict = get_uc_rotation_vpf_service(img_url)
        # angle = res_dict["data"]["angle"]

        try:
            res_dict = get_trt_rotation_vpf_service(img_url)
            print('[Info] 1223: {}'.format(img_url))
            angle = res_dict["data"]["data"]["angle"]

            angle = int(angle)
        except Exception as e:
            print('[Exception] img_url: {}'.format(img_url))
            angle = -1
        return angle

    def process_1000_items(self, is_vpf=True):
        """
        处理1000条基准数据
        """
        if is_vpf:
            print('[Info] 使用VPF模式!')
        else:
            print('[Info] 使用本地模式')
        # in_file = os.path.join(DATA_DIR, 'test_1000_res.right.e2.csv')
        in_file = os.path.join(DATA_DIR, 'test_400_res.right.e0.csv')
        data_lines = read_file(in_file)
        out_list = []
        n_old_right = 0
        n_right = 0
        n_all = 0
        n_error = 0
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            url, r_angle, dmy_angle, is_dmy, uc_angle, is_uc = data_line.split(',')

            uc_angle = int(uc_angle)
            uc_is_ok = int(is_uc)
            r_angle = int(r_angle)

            if is_vpf:
                x_angle = self.process_img_vpf(url)
            else:
                x_angle = self.process_img_url(url)

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

        out_file = os.path.join(DATA_DIR, 'check_{}_{:.4f}.e{}.{}.xlsx'
                                .format(self.model_name, safe_div(n_right, n_all), n_error, get_current_time_str()))

        write_list_to_excel(
            out_file,
            ["url", "r_angle", "dmy_angle", "is_dmy", "uc_angle", "uc_is_ok", "p_angle", "p_is_ok"],
            out_list
        )

    def evaluate_bad_cases(self, is_vpf=True):
        if is_vpf:
            print('[Info] 使用VPF模式!')
        else:
            print('[Info] 使用本地模式')

        error_file = os.path.join(DATA_DIR, 'datasets_val', 'bad_cases_urls.txt')
        out_file = os.path.join(DATA_DIR, 'datasets_val', 'bad_cases_urls_out.{}.txt'.format(get_current_time_str()))

        n_right = 0
        data_lines = read_file(error_file)
        n_all = len(data_lines)
        print('[Info] data_lines: {}'.format(n_all))

        for data_line in data_lines:
            url, angle = data_line.split(',')
            angle = int(angle)
            if is_vpf:
                x_angle = self.process_img_vpf(url)
            else:
                x_angle = self.process_img_url(url)

            x_angle = int(x_angle)
            if angle == x_angle:
                n_right += 1
                print('[Info] 正确')

            out_list = [url, str(angle), str(x_angle)]
            out_line = ",".join(out_list)
            write_line(out_file, out_line)
            print('-' * 50)

        print('[Info] 正确率: {}'.format(safe_div(n_right, n_all)))
        print('[Info] 完成! {}'.format(out_file))

    def evaluate_bad_dir(self):
        error_dir = os.path.join(DATA_DIR, 'pg_errors')
        out_dir = os.path.join(DATA_DIR, 'pg_errors_out')
        mkdir_if_not_exist(out_dir)

        paths_list, names_list = traverse_dir_files(error_dir)
        for path, name in zip(paths_list, names_list):
            angle = self.predict_img_path(path)
            img_bgr = cv2.imread(path)
            img_out = rotate_img_with_bound(img_bgr, angle)
            out_name = name.split('.')[0] + "out.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, img_out)
            print('[Info] 写入文件完成: {}'.format(out_path))

    def evaluate_pg_res(self):
        file_path = os.path.join(DATA_DIR, 'pg_dst_result.txt')
        out_dir = os.path.join(DATA_DIR, 'pg_dst_result_out')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(file_path)
        for data_line in data_lines:
            items = data_line.split('\t')
            url = items[0]
            is_ok, img_bgr = download_url_img(url)
            angle = self.predict_img_bgr(img_bgr)
            img_out = rotate_img_with_bound(img_bgr, angle)
            out_name = url.split('/')[-1]
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, img_out)
            print('[Info] 处理完成: {}'.format(out_path))
            # print(url)
            # break


def main():
    pt = ProblemTester()
    # pt.process_1000_items()
    # pt.evaluate_bad_cases()
    # pt.evaluate_bad_dir()
    pt.evaluate_pg_res()


if __name__ == '__main__':
    main()
