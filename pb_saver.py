#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 30.8.21
"""
import argparse
import os

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from myutils.project_utils import get_current_time_str, mkdir_if_not_exist
from root_dir import DATA_DIR
from utils import angle_error


class PbSaver(object):
    """
    存储PB模型
    """
    def __init__(self, model_path, out_dir):
        self.model_path = model_path
        self.out_dir = out_dir
        mkdir_if_not_exist(self.out_dir)
        print('[Info] 模型路径: {}'.format(self.model_path))
        print('[Info] 输出文件夹: {}'.format(self.out_dir))

    def save_pb_model(self, model, pb_dir, pb_name):
        """
        存储PB模型
        """
        saved_path = os.path.join(pb_dir, "{}".format(pb_name.split(".")[0]))
        model.save(saved_path)

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))

        # batch_size 16
        inputs_shape = model.inputs[0].shape
        print('[Info] inputs_shape: {}'.format(inputs_shape))
        inputs_shape_x = tf.TensorShape((1, 448, 448, 3))
        print('[Info] inputs_shape_x: {}'.format(inputs_shape_x))
        full_model = full_model.get_concrete_function(
            [tf.TensorSpec(inputs_shape_x, model.inputs[0].dtype)])

        # batch_size 1
        # full_model = full_model.get_concrete_function(
        #     [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)])

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
        frozen_out_path = pb_dir
        # name of the .pb file
        frozen_graph_filename = pb_name

        # Save frozen graph to disk
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=frozen_out_path,
                          name=frozen_graph_filename,
                          as_text=False)
        # Save its text representation
        # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
        #                   logdir=frozen_out_path,
        #                   name=f"{frozen_graph_filename}.pbtxt",
        #                   as_text=True)

        print('[Info] 存储PB模型完成! {}, {}'.format(frozen_out_path, frozen_graph_filename))

    def load_model(self, model_path, out_dir, pb_name):
        """
        加载模型
        """
        dependencies = {
            "angle_error": angle_error
        }

        model = tf.keras.models.load_model(model_path, custom_objects=dependencies, compile=False)

        self.save_pb_model(model, out_dir, pb_name)  # 存储pb模型

        return model

    def process(self):
        model_name = self.model_path.split("/")[-1].split('.')[0]
        print("[Info] 模型名称: {}".format(model_name))
        pb_name = "{}_{}.pb".format(model_name, get_current_time_str())
        print("[Info] pb模型路径: {}".format(pb_name))
        self.load_model(self.model_path, self.out_dir, pb_name)
        print("[Info] 模型存储完成: {}, pb: {}".format(self.out_dir, pb_name))


def parse_args():
    """
    处理脚本参数，支持相对路径
    img_file 文件路径，默认文件夹：img_downloader/urls
    out_folder 输出文件夹，默认文件夹：img_data
    :return: arg_img，文件路径；out_folder，输出文件夹
    """
    parser = argparse.ArgumentParser(description='转换PB模型')
    parser.add_argument('-m', dest='model', required=False, help='模型版本', type=str)
    parser.add_argument('-o', dest='outdir', required=False, help='输出文件夹', type=str,
                        default=os.path.join(DATA_DIR, "pb_models"))

    args = parser.parse_args()

    arg_model = args.model
    # arg_model = os.path.join(DATA_DIR, "models", "rotnet_resnet50_trans_best_20210830.hdf5")
    print("模型路径: {}".format(arg_model))

    arg_outdir = args.outdir
    print("输出文件夹: {}".format(arg_outdir))

    return arg_model, arg_outdir


def main():
    """
    入口函数
    """
    arg_model, arg_outdir = parse_args()
    pt = PbSaver(model_path=arg_model, out_dir=arg_outdir)
    pt.process()


if __name__ == "__main__":
    main()
