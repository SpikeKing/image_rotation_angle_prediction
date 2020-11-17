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
import tensorflow.python.keras.backend as K
import tensorflow as tf
import tensorflow_core
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
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

    @staticmethod
    def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        """
        Freezes the state of a session into a pruned computation graph.

        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                              or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices Remove the device directives from the graph for better portability.
        @return The frozen graph definition.
        """
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.compat.v1.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph


    def load_model(self):
        """
        加载模型
        """
        # model_location = os.path.join(DATA_DIR, 'models', 'problem_rotnet_resnet50.1.1177.20201115.hdf5')
        model_location = os.path.join(DATA_DIR, 'models', 'problem_rotnet_resnet50.1.5448-20201117.hdf5')
        model = load_model(model_location, custom_objects={'angle_error': angle_error})

        # sess = K.get_session()
        # frozen_graph = self.freeze_session(sess)
        # input_tensors = ['input_1:0']
        # output_tensors = ['fc360/Identity:0']

        # model.save('saved_model-1', save_format="tf")
        # print('[Info] model_location: {}'.format(model_location))
        # model = load_model(model_location, custom_objects={'angle_error': angle_error})

        # # Convert Keras model to ConcreteFunction
        # full_model = tf.function(lambda x: model(x))
        # full_model = full_model.get_concrete_function(
        #     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        # # Get frozen ConcreteFunction
        # frozen_func = convert_variables_to_constants_v2(full_model)
        # frozen_func.graph.as_graph_def()
        # layers = [op.name for op in frozen_func.graph.get_operations()]
        # print("-" * 60)
        # print("Frozen model layers: ")
        # for layer in layers:
        #     print(layer)
        # print("-" * 60)
        # print("Frozen model inputs: ")
        # print(frozen_func.inputs)
        # print("Frozen model outputs: ")
        # print(frozen_func.outputs)
        #
        # # path of the directory where you want to save your model
        # frozen_out_path = ''
        # # name of the .pb file
        # frozen_graph_filename = "frozen_graph"
        #
        # # Save frozen graph to disk
        # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
        #                   logdir=frozen_out_path,
        #                   name=f"{frozen_graph_filename}.pb",
        #                   as_text=False)
        # # Save its text representation
        # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
        #                   logdir=frozen_out_path,
        #                   name=f"{frozen_graph_filename}.pbtxt",
        #                   as_text=True)

        return model

    def predict_img(self, test_img_bgr):
        """
        预测角度
        """
        # show_img_bgr(img_bgr)
        test_img_rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)
        test_img_rgb = cv2.resize(test_img_rgb, (224, 224))
        test_img_bgr_b = np.expand_dims(test_img_rgb, axis=0)
        prediction = self.model.predict(test_img_bgr_b)
        # angle = int(K.argmax(prediction[0])) * 90
        angle = int(K.argmax(prediction[0])) % 360
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
        angle = (360 - angle) % 360
        print('[Info] img_id: {}'.format(img_id))
        print('[Info] url: {}'.format(url))
        print('[Info] angle: {}'.format(angle))
        is_ok, img_bgr = download_url_img(url)
        s_time = time.time()
        p_angle = self.predict_img(img_bgr)
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

    def process(self):
        """
        处理全部数据
        """
        in_file = os.path.join(DATA_DIR, '2020_11_12_out.9979.txt')
        out_file_format = os.path.join(DATA_DIR, 'out_file.{}.{}.xlsx')
        out_dir = os.path.join(DATA_DIR, 'out_imgs_{}'.format(get_current_time_str()))
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
            if (idx+1) == 200:
                break

        titles = ["img_id", "url", "angle", "p_angle", "abs_angle", "elapsed_time"]
        out_file = out_file_format.format(len(res_list), get_current_time_str())
        write_list_to_excel(out_file, titles, res_list)
        print('[Info] 写入文件完成! {}'.format(out_file))


def main():
    ip = ImgPredictor()
    ip.process()



if __name__ == '__main__':
    main()