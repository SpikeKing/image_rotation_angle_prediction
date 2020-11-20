from __future__ import print_function

import os
import sys
import random


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Conv1D, MaxPool1D, concatenate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator
from train.data_utils import get_problems_data
from root_dir import ROOT_DIR, DATA_DIR
from myutils.project_utils import get_current_time_str


data1_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_12w')
print('[Info] data1_path: {}'.format(data1_path))
train1_filenames, test1_filenames = get_problems_data(data1_path)
train1_filenames = (train1_filenames + test1_filenames)
random.shuffle(train1_filenames)
print('[Info] data1 train: {}, test: {}'.format(len(train1_filenames), len(test1_filenames)))

data2_path = os.path.join(ROOT_DIR, '..', 'datasets', 'biaozhu_csv_out')
print('[Info] data2_path: {}'.format(data2_path))
train2_filenames, test2_filenames = get_problems_data(data2_path)
train2_filenames = (train2_filenames + test2_filenames)
random.shuffle(train2_filenames)
print('[Info] data2 train: {}, test: {}'.format(len(train2_filenames), len(test2_filenames)))

# data3_path = os.path.join(ROOT_DIR, '..', 'datasets', 'application_3499_1024_358w')
# print('[Info] data3_path: {}'.format(data3_path))
# train3_filenames, test3_filenames = get_problems_data(data3_path)
# random.shuffle(train3_filenames)
# train3_filenames = train3_filenames[:1000000]
# test3_filenames = test3_filenames[:10000]
# print('[Info] data3 train: {}, test: {}'.format(len(train3_filenames), len(test3_filenames)))

# train_filenames = train1_filenames + train2_filenames + train3_filenames
# test_filenames = test1_filenames + test2_filenames + test3_filenames

train_filenames = train1_filenames + train2_filenames
test_filenames = test1_filenames + test2_filenames

train_filenames = train_filenames * 5

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'problem_rotnet_resnet50'

# number of classes
nb_classes = 360
# input image shape
# input_shape = (336, 336, 3)
input_shape = (224, 224, 3)
print('[Info] input_shape: {}'.format(input_shape))

# load base model
# base_model = ResNet50(weights='imagenet', include_top=False,
#                       input_shape=input_shape)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

x1 = base_model.output
x1 = Flatten()(x1)

input_ratio = Input(shape=(1, ), name='ratio')
x2 = Dense(10, activation='relu')(input_ratio)

# append classification layer
x = concatenate([x1, x2])

final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=[base_model.input, input_ratio], outputs=final_output)

# model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.9),
              metrics=[angle_error])

# training parameters
batch_size = 128
nb_epoch = 200

# 加载已有模型
# model_path = os.path.join(DATA_DIR, 'models', 'problem_rotnet_resnet50.100w-1.5339.20201115.hdf5')
# model_path = os.path.join(DATA_DIR, 'models', 'problem_rotnet_resnet50.1.7339-20201119.hdf5')
# model.load_weights(model_path)
# print('[Info] 加载模型的路径: {}'.format(model_path))

output_folder = 'models_{}_{}'.format(len(train_filenames), get_current_time_str())
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print('[Info] 模型文件夹: {}'.format(output_folder))

# callbacks
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=False,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=False,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    # callbacks=[checkpointer, reduce_lr, tensorboard],
    workers=10
)
