from __future__ import print_function

import os
import sys

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error_regression, RotNetDataGenerator
from train.data_utils import get_problems_data
from root_dir import ROOT_DIR


data_path = os.path.join(ROOT_DIR, '..', 'datasets', 'rotation_datasets_13w_512_p')
print('[Info] data_path: {}'.format(data_path))
train_filenames, test_filenames = get_problems_data(data_path)

print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')

model_name = 'problem_rotnet_resnet50_regression'

# input image shape
input_shape = (224, 224, 3)

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
    # load base model
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)

# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(1, activation='sigmoid', name='fc1')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

# model.summary()

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

optimizer = Adam(learning_rate=lr_schedule)

# model compilation
model.compile(loss=angle_error_regression,
              optimizer='adam')

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model.load_weights(os.path.join(output_folder, model_name + '.hdf5'))

# training parameters
batch_size = 128
nb_epoch = 200

# callbacks
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    save_best_only=True
)
early_stopping = EarlyStopping(patience=10)
tensorboard = TensorBoard()

# training loop
model.fit(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        one_hot=False,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        one_hot=False,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, early_stopping, tensorboard],
    # nb_worker=10,
    # pickle_safe=True,
    verbose=1
)
