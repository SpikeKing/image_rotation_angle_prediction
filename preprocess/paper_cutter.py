#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 14.11.20
"""

import os
import cv2

from root_dir import DATA_DIR, ROOT_DIR
from myutils.project_utils import *
from myutils.cv_utils import *
from utils import generate_rotated_image


class PaperCutter(object):
    def __init__(self):
        self.rotate = True
        pass

    def process_img(self, image):
        if self.rotate:
            # get a random angle
            if random_prob(0.2):
                if random_prob(0.4):
                    rotation_angle = np.random.randint(360)
                else:
                    a = np.random.randint(-5, 5)
                    b = random_pick([0, 90, 180, 270], [0.25, 0.25, 0.25, 0.25])
                    rotation_angle = abs(a + b)
            else:  # 增加边界角度计算
                if random_prob(0.5):
                    rotation_angle = 270
                else:
                    if random_prob(0.8):
                        rotation_angle = random_pick([90, 270], [0.5, 0.5])
                    else:
                        rotation_angle = random_pick([0, 180], [0.5, 0.5])
        else:
            rotation_angle = 0

        # generate the rotated image
        rotated_image, rotation_angle = generate_rotated_image(
            image,
            rotation_angle,
            size=(224, 224),
            crop_center=False,
            crop_largest_rect=True
        )

        return rotated_image, rotation_angle

    def process(self):
        # data_dir = os.path.join(DATA_DIR, 'papers', "application_3499_1024")
        data_dir = os.path.join(ROOT_DIR, '..', 'datasets', "rotation_datasets_12w")
        out_dir = os.path.join(DATA_DIR, 'papers', "application_3499_1024_out")
        mkdir_if_not_exist(out_dir)
        paths_list, names_list = traverse_dir_files(data_dir)

        random.seed(47)
        for path, name in zip(paths_list, names_list):
            print('[Info] path: {}'.format(path))
            img_name = name.split('.')[0]
            img_bgr = cv2.imread(path)
            h, w, _ = img_bgr.shape
            # for i in range(20):
            #     out_path = os.path.join(out_dir, img_name + ".out{}.jpg".format(i))
            #     if random_prob(0.5):
            #         img_out = random_crop(img_bgr, h // 10, int(w * 0.8))
            #     else:
            #         x = random.randint(2, 10)
            #         img_out = random_crop(img_bgr, h // x, int(w * 0.8))
            for i in range(20):
                rotated_image, rotation_angle = self.process_img(img_bgr)
                out_path = os.path.join(out_dir, img_name + ".out.{}.jpg".format(rotation_angle))
                cv2.imwrite(out_path, rotated_image)


def main():
    pc = PaperCutter()
    pc.process()


if __name__ == '__main__':
    main()
