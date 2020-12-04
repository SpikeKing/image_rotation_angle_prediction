#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 11.11.20
"""
import os
import sys

import cv2
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.project_utils import *
from x_utils.vpf_utils import get_uc_rotation_vpf_service

from root_dir import DATA_DIR
from utils import rotate


class ProblemTester(object):
    def __init__(self):
        self.model_name = "rotnet_v3_mobilenetv2_0.01_20201204.h5"
        print('[Info] model name: {}'.format(self.model_name))
        self.model = self.load_model()

    @staticmethod
    def save_pb_model(model):
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
             tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)])
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

    def predict_img_bgr_prob(self, img_bgr):
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_rgb_resized = cv2.resize(img_bgr, (224, 224))  # resize
        # img_rgb_resized = cv2.resize(img_bgr, (448, 448))  # resize

        img_bgr_b = np.expand_dims(img_rgb_resized, axis=0)
        img_bgr_b = preprocess_input(img_bgr_b)

        prediction = self.model.predict(img_bgr_b)
        probs = prediction[0]
        return probs

    def predict_img_bgr(self, img_bgr):
        """
        预测角度
        """
        img_bgr = rotate_img_with_bound(img_bgr, 90)
        img_bgr = rotate_img_with_bound(img_bgr, -90)
        probs = self.predict_img_bgr_prob(img_bgr)
        angle = (int(K.argmax(probs))) * 90 % 360
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

    def process_item(self, data_dict, out_dir):
        """
        处理单个数据
        """
        img_id = data_dict["id"]
        url = data_dict["url"]
        angle = data_dict["angle"]
        angle = (360 - angle) % 360
        print('[Info] img_id: {}'.format(img_id))
        print('[Info] url: {}'.format(url))
        print('[Info] angle: {}'.format(angle))
        is_ok, img_bgr = download_url_img(url)
        h, w, _ = img_bgr.shape
        s_time = time.time()
        p_angle = self.predict_img_bgr(img_bgr)
        print('[Info] p_angle: {}'.format(p_angle))

        elapsed_time = time.time() - s_time
        a = angle - p_angle
        abs_angle = abs((a + 180) % 360 - 180)

        if abs_angle > 5:
            img_bgr_d = rotate_img_with_bound(img_bgr, angle)
            img_bgr_p = rotate_img_with_bound(img_bgr, p_angle)
            img_merged = merge_two_imgs(img_bgr_d, img_bgr_p)
            out_file = os.path.join(out_dir, '{}_{}.jpg'.format(img_id, abs_angle))
            cv2.imwrite(out_file, img_merged)

        return [img_id, url, angle, p_angle, abs_angle, elapsed_time]

    def process_item_v2(self, url):
        is_ok, img_bgr = download_url_img(url)
        angle = self.predict_img_bgr(img_bgr)
        return angle

    def process_1000_items(self):
        """
        处理1000条基准数据
        """
        in_file = os.path.join(DATA_DIR, 'test_1000_res.right.e2.csv')
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

            x_angle = self.process_item_v2(url)

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

        out_file = os.path.join(DATA_DIR, 'check_{}_{}.e{}.xlsx'
                                .format(self.model_name, safe_div(n_right, n_all), n_error))

        write_list_to_excel(
            out_file,
            ["url", "r_angle", "dmy_angle", "is_dmy", "uc_angle", "uc_is_ok", "p_angle", "p_is_ok"],
            out_list
        )

    def new_test(self):
        # img_path = os.path.join(DATA_DIR, 'datasets_val', 'val_1000', 'O1CN01395Jpz1zErXatzlXa_!!6000000006683-0-quark.jpg')
        # image = cv2.imread(img_path)
        #
        # image, angle, rotated_ratio, is_ok = generate_rotated_image(
        #     image,
        #     270,
        #     size=None,
        #     crop_center=False,
        #     crop_largest_rect=True
        # )

        img_path = os.path.join(DATA_DIR, 'datasets_val', 'x2.jpg')
        image = cv2.imread(img_path)
        image = rotate(image, 90)
        image = rotate(image, -90)

        # url = "https://img.alicdn.com/imgextra/i2/6000000006683/O1CN01395Jpz1zErXatzlXa_!!6000000006683-0-quark.jpg"
        # is_ok, image = download_url_img(url)
        # show_img_bgr(image)

        show_img_bgr(image)

        x = self.predict_img_bgr(image)

        print(x)

    def process_item_v3(self, url):
        print('[Info] vpf: {}'.format(url))
        res_dict = get_uc_rotation_vpf_service(url)
        angle = res_dict['data']['angle']
        angle = int(angle)
        return angle


def demo_of_urls():
    url_path = os.path.join(DATA_DIR, 'long_text_2020-12-02-09-44-42.txt')
    out_dir = os.path.join(DATA_DIR, 'long_text_2020-12-02-09-44-42-vpf')
    mkdir_if_not_exist(out_dir)

    ip = ImgPredictor()
    for url in read_file(url_path):
        name = url.split('/')[-1].split("?")[0]
        is_ok, img_bgr = download_url_img(url)
        # angle = ip.predict_img_bgr(img_bgr)
        angle = ip.process_item_v3(url)
        img_bgr = rotate_img_for_4angle(img_bgr, angle)
        out_path = os.path.join(out_dir, "{}".format(name))
        print('out_path: {}'.format(out_path))
        cv2.imwrite(out_path, img_bgr)

    print('[Info] 处理完成: {}'.format(out_dir))


def demo_of_img_dir():
    error_dir = os.path.join(DATA_DIR, 'datasets_val', 'TestCases32')
    error2_dir = os.path.join(DATA_DIR, 'datasets_val', 'TestCases32_out')
    mkdir_if_not_exist(error2_dir)
    ip = ImgPredictor()
    paths_list, names_list = traverse_dir_files(error_dir)
    for idx, (path, name) in enumerate(zip(paths_list, names_list)):
        img_bgr = cv2.imread(path)
        angle = ip.predict_img_bgr(img_bgr)
        img_bgr = rotate_img_for_4angle(img_bgr, angle)
        out_path = os.path.join(error2_dir, "{}.jpg".format(idx))
        print('out_path: {}'.format(out_path))
        cv2.imwrite(out_path, img_bgr)
        show_img_bgr(img_bgr)
        print('-' * 50)
    print('[Info] 完成!')


def demo_of_one_img():
    name = '2.270.jpg'
    img_path = os.path.join(DATA_DIR, 'cases', name)
    r_angle = int(name.split('.')[1])
    ip = ImgPredictor()
    p_angle = ip.predict_img_path(img_path)
    print('[Info] r_angle: {}, p_angle: {}, is_equal: {}'.format(r_angle, p_angle, r_angle == p_angle))


def main():
    pt = ProblemTester()
    pt.process_1000_items()


if __name__ == '__main__':
    main()