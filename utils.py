from __future__ import division

import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from myutils.cv_utils import random_crop, rotate_img_with_bound
from myutils.project_utils import random_pick, random_prob, safe_div


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def angle_error_regression(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    return K.mean(angle_difference(y_true * 360, y_pred * 360))


def binarize_images(x):
    """
    Convert images to range 0-1 and binarize them by making
    0 the values below 0.1 and 1 the values above 0.1.
    """
    x /= 255
    x[x >= 0.1] = 1
    x[x < 0.1] = 0
    return x


def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def crop_largest_rectangle(image, angle, height, width):
    """
    Crop around the center the largest possible rectangle
    found with largest_rotated_rect.
    """
    return crop_around_center(
        image,
        *largest_rotated_rect(
            width,
            height,
            math.radians(angle)
        )
    )


def format_angle(angle):
    """
    格式化角度
    """
    angle = int(angle)
    if angle <= 45 or angle >= 325:
        r_angle = 0
    elif 45 < angle <= 135:
        r_angle = 90
    elif 135 < angle <= 225:
        r_angle = 180
    else:
        r_angle = 270
    return r_angle


def get_radio_and_resize(image, size):
    rh, rw, _ = image.shape
    rhw_ratio = safe_div(float(rh), float(rw))  # 高宽比例
    if size:
        image = cv2.resize(image, size)  # 普通的Resize
        # from myutils.cv_utils import resize_image_with_padding
        # rotated_image = resize_image_with_padding(image, desired_size=size[0])  # Padding Resize
    return rhw_ratio, image


def generate_rotated_image(image, angle, size=None, crop_center=False,
                           crop_largest_rect=False):
    """
    Generate a valid rotated image for the RotNetDataGenerator. If the
    image is rectangular, the crop_center option should be used to make
    it square. To crop out the black borders after rotation, use the
    crop_largest_rect option. To resize the final image, use the size
    option.
    """
    from myutils.cv_utils import rotate_img_for_4angle

    image_copy = image.copy()
    height, width = image.shape[:2]
    if crop_center:
        if width < height:
            height = width
        else:
            width = height

    try:
        # image = rotate(image, angle)  # 第1种旋转模式
        if angle % 90 != 0:
            rotated_image = rotate_img_with_bound(image, angle)  # 第2种旋转模型
        else:
            rotated_image = rotate_img_for_4angle(image, (360 - angle) % 360)

        if crop_largest_rect:  # 最大剪切
            rotated_image = crop_largest_rectangle(rotated_image, angle, height, width)
        rhw_ratio, rotated_image = get_radio_and_resize(rotated_image, size)
    except Exception as e:
        angle = format_angle(angle)
        rotated_image = rotate_img_for_4angle(image_copy, (360 - angle) % 360)
        rhw_ratio, rotated_image = get_radio_and_resize(rotated_image, size)

    return rotated_image, angle, rhw_ratio


class RotNetDataGenerator(Iterator):
    """
    Given a NumPy array of images or a list of image paths,
    generate batches of rotated images and rotation angles on-the-fly.
    """

    def __init__(self, input, input_shape=None, color_mode='rgb', batch_size=64,
                 one_hot=True, preprocess_func=None, rotate=True, crop_center=False,
                 crop_largest_rect=False, shuffle=False, seed=None,
                 is_train=True,
                 is_hw_ratio=False,
                 is_random_crop=False,
                 random_angle=8,
                 nb_classes=4):

        self.images = None
        self.filenames = None
        self.input_shape = input_shape
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.preprocess_func = preprocess_func
        self.rotate = rotate
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect
        self.shuffle = shuffle

        # 新增参数
        self.is_train = is_train  # 是否增强摆动数据
        self.is_hw_ratio = is_hw_ratio  # 是否增加高宽比例
        self.is_random_crop = is_random_crop  # 是否支持高度随机剪裁
        self.random_angle = random_angle
        self.nb_classes = nb_classes

        print('[Info] ' + "-" * 50)
        print('[Info] 数据参数: ')
        print('[Info] color_mode: {}'.format(self.color_mode))
        print('[Info] is_train: {}'.format(self.is_train))
        print('[Info] is_hw_ratio: {}'.format(self.is_hw_ratio))
        print('[Info] random_angle: {}'.format(self.random_angle))
        print('[Info] nb_classes: {}'.format(self.nb_classes))
        print('[Info] is_random_crop: {}'.format(self.is_random_crop))
        print('[Info] ' + "-" * 50)

        if self.color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', self.color_mode, '; expected "rgb" or "grayscale".')

        # check whether the input is a NumPy array or a list of paths
        if isinstance(input, np.ndarray):
            self.images = input
            N = self.images.shape[0]
            if not self.input_shape:
                self.input_shape = self.images.shape[1:]
                # add dimension if the images are greyscale
                if len(self.input_shape) == 2:
                    self.input_shape = self.input_shape + (1,)
        else:
            self.filenames = input
            N = len(self.filenames)

        super(RotNetDataGenerator, self).__init__(N, batch_size, shuffle, seed)

    @staticmethod
    def center_crop_by_hw(img_bgr):
        """
        避免图像的比例失衡
        """
        h, w, _ = img_bgr.shape
        if h // w > 4:
            mid = h // 2
            img_crop = img_bgr[mid - 2 * w:mid + 2 * w, :, :]
            return img_crop
        if w // h > 4:
            mid = w // 2
            img_crop = img_bgr[:, mid - 2 * h:mid + 2 * h, :]
            return img_crop
        else:
            return img_bgr

    def process_img(self, image):
        if self.rotate:  # 随机旋转
            if self.is_train and random_prob(0.5):
                offset_angle = random.randint(self.random_angle*(-1), self.random_angle)
            else:
                offset_angle = 0
            rotation_angle = random_pick([0, 90, 180, 270], [0.25, 0.25, 0.25, 0.25])
            rotation_angle = (rotation_angle + offset_angle) % 360
        else:
            rotation_angle = 0

        image = RotNetDataGenerator.center_crop_by_hw(image)

        # generate the rotated image
        image, angle, rhw_ratio = generate_rotated_image(
            image,
            rotation_angle,
            size=self.input_shape[:2],
            crop_center=self.crop_center,
            crop_largest_rect=self.crop_largest_rect
        )

        angle = format_angle(angle)  # 输出固定的度数

        return image, angle, rhw_ratio

    def _get_batches_of_transformed_samples(self, index_array):
        # create array to hold the images
        batch_x = np.zeros((len(index_array),) + self.input_shape, dtype='float32')
        batch_x_2 = np.zeros(len(index_array), dtype='float32')

        # create array to hold the labels
        batch_y = np.zeros(len(index_array), dtype='float32')

        img_list, ratio_list, angle_list = [], [], []  # 图像, 比例, 角度
        # iterate through the current batch
        for i, j in enumerate(index_array):
            if self.filenames is None:
                image = self.images[j]
            else:
                is_color = int(self.color_mode == 'rgb')
                try:
                    image = cv2.imread(self.filenames[j], is_color)
                    try:
                        if is_color:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        # print('[Warning] 兼容模式 image path: {}'.format(self.filenames[j]))
                        from myutils.img_compat import ImgCompatBGR
                        image = ImgCompatBGR.imread(self.filenames[j])
                        if is_color:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print('[Error] e: {}'.format(e))
                    print('[Error] 错误 image path: {}'.format(self.filenames[j]))
                    continue

                # 随机剪裁
                if self.is_train and self.is_random_crop:
                    if random_prob(0.5):
                        if random_prob(0.5):
                            h, w, _ = image.shape
                            out_h = int(h // 4 * 3)  # mode 1
                            image = random_crop(image, out_h, w)
                        else:
                            h, w, _ = image.shape
                            out_w = int(w // 4 * 3)  # mode 1
                            image = random_crop(image, h, out_w)
            try:
                rotated_image, rotation_angle, rotated_ratio = self.process_img(image)
            except Exception as e:
                print('[Error] e: {}'.format(e))
                print('[Error] 错误 image path: {}'.format(self.filenames[j]))
                continue

            # add dimension to account for the channels if the image is greyscale
            if rotated_image.ndim == 2:
                rotated_image = np.expand_dims(rotated_image, axis=2)

            if self.nb_classes == 4:
                rotation_angle_idx = format_angle(rotation_angle) // 90
            elif self.nb_classes == 360:
                rotation_angle_idx = rotation_angle
            else:
                raise Exception("[Exception] 角度 {} 支持 4(0-90-180-270) 或 360!".format(self.nb_classes))

            # 测试
            # print('[Test] rotated_image: {}, rotated_ratio: {}, rotation_angle_idx: {}'.format(
            #     rotated_image.shape, rotated_ratio, rotation_angle_idx))

            # store the image and label in their corresponding batches
            batch_x[i] = rotated_image
            batch_x_2[i] = rotated_ratio
            batch_y[i] = rotation_angle_idx

        if self.one_hot:
            # convert the numerical labels to binary labels
            batch_y = to_categorical(batch_y, self.nb_classes)
        else:
            batch_y /= 360

        # preprocess input images
        if self.preprocess_func:
            batch_x = self.preprocess_func(batch_x)

        if self.is_hw_ratio:
            return [batch_x, batch_x_2], batch_y
        else:
            return batch_x, batch_y

    def next(self):
        with self.lock:
            # get input data index and size of the current batch
            index_array = next(self.index_generator)
        # create array to hold the images
        return self._get_batches_of_transformed_samples(index_array)


def display_examples(model, input, num_images=5, size=None, crop_center=False,
                     crop_largest_rect=False, preprocess_func=None, save_path=None):
    """
    Given a model that predicts the rotation angle of an image,
    and a NumPy array of images or a list of image paths, display
    the specified number of example images in three columns:
    Original, Rotated and Corrected.
    """

    if isinstance(input, (np.ndarray)):
        images = input
        N, h, w = images.shape[:3]
        if not size:
            size = (h, w)
        indexes = np.random.choice(N, num_images)
        images = images[indexes, ...]
    else:
        images = []
        filenames = input
        N = len(filenames)
        indexes = np.random.choice(N, num_images)
        for i in indexes:
            image = cv2.imread(filenames[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.asarray(images)

    x = []
    y = []
    for image in images:
        rotation_angle = np.random.randint(360)
        rotated_image = generate_rotated_image(
            image,
            rotation_angle,
            size=size,
            crop_center=crop_center,
            crop_largest_rect=crop_largest_rect
        )
        x.append(rotated_image)
        y.append(rotation_angle)

    x = np.asarray(x, dtype='float32')
    y = np.asarray(y, dtype='float32')

    if x.ndim == 3:
        x = np.expand_dims(x, axis=3)

    y = to_categorical(y, 360)

    x_rot = np.copy(x)

    if preprocess_func:
        x = preprocess_func(x)

    y = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(x), axis=1)

    plt.figure(figsize=(10.0, 2 * num_images))

    title_fontdict = {
        'fontsize': 14,
        'fontweight': 'bold'
    }

    fig_number = 0
    for rotated_image, true_angle, predicted_angle in zip(x_rot, y, y_pred):
        original_image = rotate(rotated_image, -true_angle)
        if crop_largest_rect:
            original_image = crop_largest_rectangle(original_image, -true_angle, *size)

        corrected_image = rotate(rotated_image, -predicted_angle)
        if crop_largest_rect:
            corrected_image = crop_largest_rectangle(corrected_image, -predicted_angle, *size)

        if x.shape[3] == 1:
            options = {'cmap': 'gray'}
        else:
            options = {}

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        if fig_number == 1:
            plt.title('Original\n', fontdict=title_fontdict)
        plt.imshow(np.squeeze(original_image).astype('uint8'), **options)
        plt.axis('off')

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        if fig_number == 2:
            plt.title('Rotated\n', fontdict=title_fontdict)
        ax.text(
            0.5, 1.03, 'Angle: {0}'.format(true_angle),
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=11
        )
        plt.imshow(np.squeeze(rotated_image).astype('uint8'), **options)
        plt.axis('off')

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        corrected_angle = angle_difference(predicted_angle, true_angle)
        if fig_number == 3:
            plt.title('Corrected\n', fontdict=title_fontdict)
        ax.text(
            0.5, 1.03, 'Angle: {0}'.format(corrected_angle),
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=11
        )
        plt.imshow(np.squeeze(corrected_image).astype('uint8'), **options)
        plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if save_path:
        plt.savefig(save_path)


def main():
    # print(-1 % 360)
    pass


if __name__ == '__main__':
    main()