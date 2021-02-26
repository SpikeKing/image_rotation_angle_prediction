#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 2.12.20
"""

import os
import random
import sys

# 增加使用GPU
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

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
from myutils.project_utils import traverse_dir_files, get_current_time_str, mkdir_if_not_exist
from root_dir import ROOT_DIR, DATA_DIR


class ProblemTrainer(object):
    def __init__(self,
                 mode="resnet50",  # 训练模式, 支持mobilenetv2和resnet50
                 nb_classes=4,
                 random_angle=8,  # 随机10度
                 is_hw_ratio=False,  # 是否使用高宽比
                 nb_epoch=200,
                 is_random_crop=True  # 随机高度剪裁
                 ):

        self.mode = mode  # 训练模式
        self.nb_classes = nb_classes  # 类别数
        self.input_shape = (448, 448, 3)  # 输入图像尺寸
        self.random_angle = random_angle  # 随机角度
        self.is_hw_ratio = is_hw_ratio  # 是否使用高宽比
        self.nb_epoch = nb_epoch  # epoch
        self.is_random_crop = is_random_crop  # 随机高度剪裁

        self.model_path = None
        if self.mode == "mobilenetv2":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_mobilenetv2_224_20201213.2.hdf5')
        elif self.mode == "resnet50v2":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_resnet50v2_448_20201216.6.hdf5')
        elif self.mode == "resnet50":
            self.model_path = os.path.join(DATA_DIR, 'models', 'rotnet_v3_resnet50_best_20210226.1.hdf5')

        if mode == "mobilenetv2":
            self.batch_size = 64  # batch size, v100
        elif mode == "resnet50":
            self.batch_size = 32  # batch size, v100
        else:
            self.batch_size = 100

        # 输出文件夹
        self.output_dir = "model_{}_{}_{}".format(self.mode, self.input_shape[0], get_current_time_str())

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
        print('[Info] ' + "-" * 50)

        self.model_name, self.model = self.init_model(self.mode)  # 初始化模型
        self.train_data, self.test_data = self.load_train_and_test_dataset()  # 加载训练和测试数据

        mkdir_if_not_exist(self.output_dir)

    def init_model(self, mode="resnet50"):
        """
        初始化模型
        :param mode: 模型类型
        :return: 模型名称和基础模型
        """
        if mode == "resnet50v2":
            from tensorflow.keras.applications.resnet_v2 import ResNet50V2
            model_name = 'rotnet_v3_resnet50v2_{epoch:02d}_{val_loss:.4f}.hdf5'
            base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif mode == "resnet50":
            from tensorflow.keras.applications.resnet import ResNet50
            model_name = 'rotnet_v3_resnet50_{epoch:02d}_{val_loss:.4f}.hdf5'
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif mode == "mobilenetv2":
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            model_name = 'rotnet_v3_mobilenetv2_{epoch:02d}_{val_loss:.4f}.hdf5'
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise Exception("[Exception] mode {} 不支持!!".format(mode))

        # freeze
        for layer in base_model.layers:
            layer.trainable = True

        x = base_model.output
        if mode == "mobilenetv2":
            x = Dense(128, activation="relu")(x)
        x = Flatten()(x)

        if self.is_hw_ratio:  # 是否使用宽高比
            x1 = base_model.output
            x1 = Flatten()(x1)
            input_ratio = Input(shape=(1,), name='ratio')
            x2 = Dense(1, activation='relu')(input_ratio)
            x = concatenate([x1, x2])

        final_output = Dense(self.nb_classes, activation='softmax', name='fc360')(x)
        model = Model(inputs=base_model.input, outputs=final_output)

        model.summary()

        # 优化器
        if self.nb_classes == 360:
            metrics = ["acc", angle_error]
        else:
            metrics = ["acc"]

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001, momentum=0.9),
                      metrics=metrics)
        if self.model_path:
            model.load_weights(self.model_path)
            print('[Info] 加载模型的路径: {}'.format(self.model_path))

        return model_name, model

    def load_train_and_test_dataset(self):
        # 9w query数据
        dataset1_path = "/ProjectRoot/workspace/problems-segmentation-yolov5/mydata/ps_datasets_v2/images"
        train1_filenames, test1_filenames = self.get_split_datasets(dataset1_path)

        # 14w query数据
        dataset2_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked_r')
        train2_filenames, test2_filenames = self.get_split_datasets(dataset2_path)

        # 5k 题库数据
        dataset3_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_tiku_5k')
        train3_filenames, test3_filenames = self.get_split_datasets(dataset3_path)

        # 2.5w 题库数据
        dataset4_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_zhengye_3w')
        train4_filenames, test4_filenames = self.get_split_datasets(dataset4_path)

        # 4w 手写数据
        dataset5_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_write_4w')
        train5_filenames, test5_filenames = self.get_split_datasets(dataset5_path)

        # 3w Query数据
        dataset6_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_ds_write2_3w')
        train6_filenames, test6_filenames = self.get_split_datasets(dataset6_path)

        dataset_val_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_val')
        test_val_filenames = self.get_total_datasets(dataset_val_path)

        # 全部数据集
        train_filenames = train1_filenames + train2_filenames + train3_filenames + \
                          train4_filenames + train5_filenames + train6_filenames

        train_filenames = train_filenames * 2

        test_filenames = test1_filenames + test2_filenames + test3_filenames + \
                         test4_filenames + test5_filenames + test6_filenames + \
                         test_val_filenames

        random.shuffle(train_filenames)
        random.shuffle(test_filenames)
        print('[Info] 训练数据: {}, 验证数据: {}'.format(len(train_filenames), len(test_filenames)))

        return train_filenames, test_filenames

    @staticmethod
    def get_split_datasets(img_dir, prob=0.95):
        """
        获取训练和测试数据
        """
        image_paths, _ = traverse_dir_files(img_dir, is_sorted=False, ext="jpg")
        random.shuffle(image_paths)

        n_train_samples = int(len(image_paths) * prob)  # 数据总量
        train_filenames = image_paths[:n_train_samples]
        test_filenames = image_paths[n_train_samples:]
        print('[Info] ' + '-' * 50)
        print('[Info] img_dir: {}'.format(img_dir))
        print('[Info] 数据总量: {}, 训练集: {}, 验证集: {}'
              .format(n_train_samples, len(train_filenames), len(test_filenames)))
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

        steps_per_epoch = len(self.train_data) // self.batch_size
        validation_steps = len(self.test_data) // self.batch_size

        monitor = 'val_acc'
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.output_dir, self.model_name),
            monitor=monitor,
            save_best_only=True
        )
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        tensorboard = TensorBoard()

        n_workers = 3
        print('[Info] n_workers: {}'.format(n_workers))

        # training loop
        self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.nb_epoch,
            validation_data=test_generator,
            validation_steps=validation_steps,
            # callbacks=[checkpointer, reduce_lr, tensorboard],
            # callbacks=[checkpointer, reduce_lr],
            callbacks=[checkpointer],
            workers=n_workers,
        )


def main():
    pt = ProblemTrainer()
    pt.train()


if __name__ == '__main__':
    main()
