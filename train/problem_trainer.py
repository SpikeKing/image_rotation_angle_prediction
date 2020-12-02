#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 2.12.20
"""

import os
import sys
import random
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from utils import angle_error, RotNetDataGenerator
from myutils.project_utils import traverse_dir_files, get_current_time_str, mkdir_if_not_exist
from root_dir import ROOT_DIR


class ProblemTrainer(object):
    def __init__(self,
                 mode="mobilenetv2",  # 训练模式, 支持mobilenetv2和resnet50
                 nb_classes=4,
                 input_shape=(224, 224, 3),  # 训练模式，支持224x224x3和448x448x3
                 batch_size=192,  # V100, 224->192, 448->48
                 nb_epoch=200,
                 ):

        self.mode = mode  # 训练模式
        self.nb_classes = nb_classes
        self.input_shape = input_shape  # 输入图像尺寸
        self.batch_size = batch_size  # batch size
        self.nb_epoch = nb_epoch  # epoch
        # 输出文件夹
        self.output_dir = "model_{}_{}_{}".format(self.mode, self.input_shape[0], get_current_time_str())

        print('[Info] ' + "-" * 50)
        print('[Info] 训练参数: ')
        print('[Info] mode: {}'.format(self.mode))
        print('[Info] nb_classes: {}'.format(self.nb_classes))
        print('[Info] input_shape: {}'.format(self.input_shape))
        print('[Info] batch_size: {}'.format(self.batch_size))
        print('[Info] nb_epoch: {}'.format(self.nb_epoch))
        print('[Info] output_dir: {}'.format(self.output_dir))
        print('[Info] ' + "-" * 50)

        _, self.model = self.init_model()  # 初始化模型
        self.train_data, self.test_data = self.load_train_and_test_dataset()  # 加载训练和测试数据

        mkdir_if_not_exist(self.output_dir)

    def init_model(self, mode="resnet50"):
        """
        初始化模型
        :param mode: 模型类型
        :return: 模型名称和基础模型
        """
        if mode == "resnet50":
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            model_name = 'rotnet_v3_resnet50'
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif mode == "mobilenetv2":
            from tensorflow.keras.applications.resnet50 import ResNet50
            model_name = 'rotnet_v3_mobilenetv2'
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise Exception("[Exception] mode {} 不支持!!".format(mode))

        x = base_model.output
        x = Flatten()(x)

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

        return model_name, model

    def load_train_and_test_dataset(self):
        # 18w有黑边的数据集
        dataset1_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v3_187281')
        train1_filenames = self.get_total_datasets(dataset1_path)

        # 14w无黑边数据
        dataset2_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked')
        train2_filenames, test2_filenames = self.get_split_datasets(dataset2_path)

        # 1k纯粹验证集
        dataset_val_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_val')
        test_val_filenames = self.get_total_datasets(dataset_val_path)

        # 全部数据集
        train_filenames = train1_filenames + train2_filenames
        test_filenames = test2_filenames + test_val_filenames * 10

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
    def get_total_datasets(img_dir):
        """
        获取全部数据
        """
        image_paths, _ = traverse_dir_files(img_dir, is_sorted=False, ext="jpg")
        random.shuffle(image_paths)

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
                shuffle=True
            )

        test_generator = RotNetDataGenerator(
                self.test_data,
                input_shape=self.input_shape,
                batch_size=self.batch_size,
                preprocess_func=preprocess_input,
                crop_center=False,
                crop_largest_rect=True,
                is_train=False  # 关闭训练参数
            )

        steps_per_epoch = len(self.train_data) // self.batch_size
        validation_steps = len(self.test_data) // self.batch_size

        monitor = 'val_acc'
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.output_dir, "model_{epoch:02d}.h5"),
            monitor=monitor,
            save_best_only=True
        )
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        tensorboard = TensorBoard()

        n_workers = 10
        print('[Info] n_workers: {}'.format(n_workers))

        # training loop
        self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.nb_epoch,
            validation_data=test_generator,
            validation_steps=validation_steps,
            callbacks=[checkpointer, reduce_lr, tensorboard],
            workers=n_workers,
        )


def main():
    pt = ProblemTrainer()
    pt.train()


if __name__ == '__main__':
    main()
