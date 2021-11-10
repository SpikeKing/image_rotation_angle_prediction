#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 2.12.20
"""
import argparse
import os
import random
import sys

# 增加使用GPU
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.visible_device_list = '0'
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import concatenate

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from utils import angle_error, RotNetDataGenerator
from myutils.project_utils import *
from root_dir import ROOT_DIR, DATA_DIR


class ProblemTrainer(object):
    def __init__(self,
                 mode="resnet50",  # 训练模式, 支持mobilenetv2和resnet50
                 file_path="",
                 number=-1,
                 nb_classes=4,
                 random_angle=10,  # 随机10度
                 is_hw_ratio=False,  # 是否使用高宽比
                 nb_epoch=10000,
                 is_random_crop=True,  # 随机高度和宽度剪裁
                 version="v1",  # 版本
                 batch_size=32,  # batch_size
                 img_size=448,  # 图像尺寸
                 ):

        self.mode = mode  # 训练模式
        self.nb_classes = nb_classes  # 类别数
        self.input_shape = (img_size, img_size, 3)  # 输入图像尺寸
        self.random_angle = random_angle  # 随机角度
        self.is_hw_ratio = is_hw_ratio  # 是否使用高宽比
        self.nb_epoch = nb_epoch  # epoch
        self.is_random_crop = is_random_crop  # 随机高度剪裁
        self.batch_size = int(batch_size)  # batch size
        self.img_size = img_size  # 图像尺寸

        self.version = version
        self.file_path = file_path
        self.sample_num = number

        self.model_path = None
        if self.mode == "mobilenetv2":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_mobilenetv2_best_20211101_tl_224.hdf5')
        elif self.mode == "resnet50v2":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_resnet50v2_448_20201216.6.hdf5')
        elif self.mode == "resnet50":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_resnet50_best_20211108.hdf5')
        elif self.mode == "densenet121":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_densenet121_best_20210914.hdf5')

        if self.batch_size <= 0 or not self.batch_size:
            if mode == "mobilenetv2":
                self.batch_size = 64  # batch size, v100
            elif mode == "resnet50" or mode == "densenet121":
                # self.batch_size = 32  # batch size, v100
                self.batch_size = 16  # batch size, v2080
            else:
                self.batch_size = 100
        print('[Info] batch_size: {}'.format(self.batch_size))

        model_folder = "model_{}_{}_{}_{}".format(self.version, self.mode, self.input_shape[0], get_current_time_str())
        self.output_dir = os.path.join(DATA_DIR, "models", model_folder)  # 输出文件夹
        mkdir_if_not_exist(self.output_dir)

        print('[Info] ' + "-" * 50)
        print('[Info] 训练参数: ')
        print('[Info] mode: {}'.format(self.mode))
        print('[Info] nb_classes: {}'.format(self.nb_classes))
        print('[Info] input_shape: {}'.format(self.input_shape))
        print('[Info] batch_size: {}'.format(self.batch_size))
        print('[Info] nb_epoch: {}'.format(self.nb_epoch))
        print('[Info] random_angle: {}'.format(self.random_angle))
        print('[Info] is_random_crop_h: {}'.format(self.is_random_crop))
        print('[Info] output_dir: {}'.format(self.output_dir))
        print('[Info] version: {}'.format(self.version))
        print('[Info] sample_num: {}'.format(self.sample_num))
        print('[Info] ' + "-" * 50)

        self.model_name, self.model = self.init_model(self.mode)  # 初始化模型

        if self.version == "v1":
            if self.file_path:
                all_data_path = self.file_path
                self.train_data, self.test_data = \
                    self.load_train_and_test_dataset_quick(all_data_path, is_val=False, num=self.sample_num)
            else:
                all_data_path = os.path.join(DATA_DIR, "files_v2", "angle_dataset_all_20211026")
                print('[Info] 样本数据汇总路径: {}'.format(all_data_path))
                self.train_data, self.test_data = \
                    self.load_train_and_test_dataset_quick(all_data_path, is_val=True, num=self.sample_num)

    def init_model(self, mode="resnet50"):
        """
        初始化模型
        :param mode: 模型类型
        :return: 模型名称和基础模型
        """
        if mode == "resnet50v2":
            from tensorflow.keras.applications.resnet_v2 import ResNet50V2
            model_name = 'rotnet_v3_resnet50v2_{epoch:02d}_{val_acc:.4f}.hdf5'
            base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif mode == "resnet50":
            from tensorflow.keras.applications.resnet import ResNet50
            model_name = 'rotnet_v3_resnet50_{epoch:02d}_{val_acc:.4f}.hdf5'
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif mode == "mobilenetv2":
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            model_name = 'rotnet_v3_mobilenetv2_{epoch:02d}_{val_acc:.4f}.hdf5'
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif mode == "densenet121":
            from tensorflow.python.keras.applications.densenet import DenseNet121
            model_name = 'rotnet_v3_densenet121_{epoch:02d}_{val_acc:.4f}.hdf5'
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise Exception("[Exception] mode {} 不支持!!".format(mode))

        # freeze
        for layer in base_model.layers:
            layer.trainable = True

        x = base_model.output
        # if mode == "mobilenetv2":
        #     x = Dense(128, activation="relu")(x)
        x = Flatten()(x)

        if self.is_hw_ratio:  # 是否使用宽高比
            x1 = base_model.output
            x1 = Flatten()(x1)
            input_ratio = Input(shape=(1,), name='ratio')
            x2 = Dense(1, activation='relu')(input_ratio)
            x = concatenate([x1, x2])

        final_output = Dense(self.nb_classes, activation='softmax', name='fc360')(x)
        model = Model(inputs=base_model.input, outputs=final_output)

        # model.summary()

        # 优化器
        if self.nb_classes == 360:
            metrics = ["acc", angle_error]
        else:
            metrics = ["acc"]

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.0004, momentum=0.9),
                      metrics=metrics)
        if self.model_path:
            model.load_weights(self.model_path)
            print('[Info] 加载模型的路径: {}'.format(self.model_path))

        return model_name, model

    @staticmethod
    def load_train_and_test_dataset_quick(path, prob=0.97, is_val=False, num=-1):
        if path.endswith("txt"):  # 直接是文件
            print('[Info] 读取文件: {}'.format(path))
            image_paths = read_file(path)
        else:  # 文件夹
            print('[Info] 读取文件夹: {}'.format(path))
            file_paths, _ = traverse_dir_files(path)
            image_paths = []
            for file in file_paths:
                sub_lines = read_file(file)
                image_paths += sub_lines
        print('[Info] 样本数: {}'.format(len(image_paths)))

        if is_val:  # 加载验证
            dataset_val_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_val')
            test_val_filenames = ProblemTrainer.get_total_datasets(dataset_val_path)
            test_val_filenames = test_val_filenames * 4  # 扩大4倍
        else:
            test_val_filenames = []

        random.seed(47)
        random.shuffle(image_paths)
        if num > 0:
            image_paths = image_paths[:num]  # 只用40万

        n_train_samples = int(len(image_paths) * prob)  # 数据总量
        train_filenames = test_val_filenames + image_paths[:n_train_samples]
        test_filenames = test_val_filenames + image_paths[n_train_samples:]

        # train_filenames = train_filenames + test_filenames
        train_filenames = train_filenames
        print('[Info] ' + '-' * 50)
        print('[Info] 数据总量: {}, 训练集: {}, 验证集: {}'.format(n_train_samples, len(train_filenames), len(test_filenames)))
        print('[Info] ' + '-' * 50)
        return train_filenames, test_filenames

    @staticmethod
    def get_total_datasets(img_dir, sample_ratio=1.0):
        """
        获取全部数据
        """
        image_paths, _ = traverse_dir_files(img_dir, is_sorted=False, ext="jpg")
        random.shuffle(image_paths)

        if sample_ratio < 1.0:
            n_paths = len(image_paths)
            rn_paths = int(n_paths * sample_ratio)
            image_paths = image_paths[:rn_paths]

        print('[Info] ' + '-' * 50)
        print('[Info] img_dir: {}'.format(img_dir))
        print('[Info] 数据总量: {}'.format(len(image_paths)))
        print('[Info] ' + '-' * 50)

        return image_paths

    def train(self):
        train_generator = RotNetDataGenerator(
            self.train_data,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            preprocess_func=preprocess_input,
            crop_center=False,
            crop_largest_rect=True,
            shuffle=True,
            is_hw_ratio=self.is_hw_ratio,
            random_angle=self.random_angle,
            is_random_crop=self.is_random_crop
        )

        test_generator = RotNetDataGenerator(
            self.test_data,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            preprocess_func=preprocess_input,
            crop_center=False,
            crop_largest_rect=True,
            is_train=False,  # 关闭训练参数
            is_hw_ratio=self.is_hw_ratio,
            random_angle=self.random_angle,
            is_random_crop=self.is_random_crop
        )

        steps_per_epoch = min(len(self.train_data) // self.batch_size, 50000)
        validation_steps = min(len(self.test_data) // self.batch_size, 5000)

        monitor = 'val_acc'
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.output_dir, self.model_name),
            monitor=monitor,
            save_best_only=True
        )
        n_workers = 3
        print('[Info] n_workers: {}'.format(n_workers))

        # training loop
        self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.nb_epoch,
            validation_data=test_generator,
            validation_steps=validation_steps,
            callbacks=[checkpointer],
            workers=n_workers,
        )


def parse_args():
    """
    处理脚本参数，支持相对路径
    img_file 文件路径，默认文件夹：img_downloader/urls
    out_folder 输出文件夹，默认文件夹：img_data
    :return: arg_img，文件路径；out_folder，输出文件夹
    """
    parser = argparse.ArgumentParser(description='训练数据')
    parser.add_argument('-v', dest='version', required=True, help='模型版本', type=str)
    parser.add_argument('-f', dest='file_path', required=False, help='数据路径', type=str)
    parser.add_argument('-b', dest='batch_size', required=False, default="16", help='模型版本', type=str)
    parser.add_argument('-s', dest='image_size', required=False, default=448, help='图像尺寸', type=int)
    parser.add_argument('-m', dest='mode', required=False, default="resnet50", help='模型版本', type=str)
    parser.add_argument('-n', dest='number', required=False, default="-1", help='样本数量', type=str)

    args = parser.parse_args()

    arg_version = args.version
    print("[Info] version: {}".format(arg_version))

    arg_file_path = args.file_path
    print("[Info] file_path: {}".format(arg_file_path))

    arg_batch_size = int(args.batch_size)
    print("[Info] batch_size: {}".format(arg_batch_size))

    arg_image_size = int(args.image_size)
    print("[Info] image_size: {}".format(arg_image_size))

    arg_mode = args.mode
    print("[Info] mode: {}".format(arg_mode))

    arg_number = int(args.number)
    print("[Info] number: {}".format(arg_number))

    return arg_version, arg_file_path, arg_batch_size, arg_image_size, arg_mode, arg_number


def main():
    """
    入口函数
    """
    arg_version, arg_file_path, arg_batch_size, arg_image_size, arg_mode, arg_number = parse_args()
    pt = ProblemTrainer(version=arg_version, file_path=arg_file_path, batch_size=arg_batch_size,
                        img_size=arg_image_size, mode=arg_mode, number=arg_number)
    pt.train()


if __name__ == '__main__':
    main()
