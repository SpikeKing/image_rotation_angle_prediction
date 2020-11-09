#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 9.11.20
"""

from myutils.project_utils import traverse_dir_files


def get_problems_data(img_dir):
    image_paths, image_names = traverse_dir_files(img_dir)

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.95)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    return train_filenames, test_filenames