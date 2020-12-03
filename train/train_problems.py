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
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator
from train.data_utils import get_problems_data
from root_dir import ROOT_DIR, DATA_DIR
from myutils.project_utils import get_current_time_str


# 包含黑边的数据
# data3_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v3_187281')
# print('[Info] data3_path: {}'.format(data3_path))
# train3_filenames, test3_filenames = get_problems_data(data3_path)
# print('[Info] data3 train: {}, test: {}'.format(len(train3_filenames), len(test3_filenames)))
# random.shuffle(train3_filenames)
# train3_filenames = train3_filenames[:len(train3_filenames)//2]

# 无黑边数据 8.8w数据
data5_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_v4_checked_r')
print('[Info] data5_path: {}'.format(data5_path))
train5_filenames, test5_filenames = get_problems_data(data5_path)
print('[Info] data5 train: {}, test: {}'.format(len(train5_filenames), len(test5_filenames)))
train5_filenames = train5_filenames * 5

# 无黑边验证数据 少量
data_val_path = os.path.join(ROOT_DIR, '..', 'datasets', 'datasets_val')
print('[Info] data_val_path: {}'.format(data_val_path))
train_val_filenames, test_val_filenames = get_problems_data(data_val_path)
test_val_filenames = train_val_filenames + test_val_filenames  # 全部数据都是验证数据
print('[Info] data val test: {}'.format(len(test_val_filenames)))

train_filenames = train5_filenames
test_filenames = test5_filenames + test_val_filenames * 10

random.shuffle(train_filenames)
random.shuffle(test_filenames)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

# model_name = 'problem_rotnet_resnet50'
model_name = 'problem_rotnet_mobilenetv2'

# number of classes
nb_classes = 360

# input image shape
input_shape = (224, 224, 3)
# input_shape = (448, 448, 3)
print('[Info] input_shape: {}'.format(input_shape))

# load base model
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

x1 = base_model.output
x1 = Flatten()(x1)

input_ratio = Input(shape=(1,), name='ratio')
x2 = Dense(10, activation='relu')(input_ratio)

# append classification layer
x = concatenate([x1, x2])
# x = x1

final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=[base_model.input, input_ratio], outputs=final_output)
# model = Model(inputs=base_model.input, outputs=final_output)

# model.summary()

# model compilation
# lr_schedule = ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=10000,
#     decay_rate=0.9
# )
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(learning_rate=lr_schedule),
#               metrics=[angle_error])
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.9),
              metrics=["acc", angle_error])

# training parameters
# batch_size = 48
batch_size = 192
nb_epoch = 200

# 加载已有模型
# model_path = os.path.join(DATA_DIR, 'models', 'problem_rotnet_mobilenetv2_448_20201201_tmp.1.hdf5')  # 最好模型
# model_path = os.path.join(DATA_DIR, 'models', 'problem_rotnet_mobilenetv2_pad448_20201201.4.hdf5')  # 最好模型
# model_path = os.path.join(DATA_DIR, 'models', 'problem_rotnet_mobilenetv2_224_20201201.4.hdf5')  # 最好模型
model_path = os.path.join(DATA_DIR, 'models', 'problem_rotnet_mobilenetv2_224_20201202.4.hdf5')  # 最好模型
model.load_weights(model_path)
print('[Info] 加载模型的路径: {}'.format(model_path))

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
# early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

# training loop
model.fit(
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
        crop_largest_rect=True,
        is_train=False
    ),
    validation_steps=len(test_filenames) / batch_size,
    # callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    callbacks=[checkpointer, reduce_lr, tensorboard],
    # callbacks=[checkpointer, tensorboard],
    workers=10,
)
