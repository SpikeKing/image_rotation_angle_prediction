#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 24.9.21
"""

from myutils.cv_utils import *
from myutils.project_utils import *


def center_crop_by_hw(img_bgr):
    """
    避免图像的比例失衡
    """
    h, w, _ = img_bgr.shape
    if h // w > 4:
        mid = h // 2
        img_crop = img_bgr[mid-2*w:mid+2*w, :, :]
        return img_crop
    if w // h > 4:
        mid = w // 2
        img_crop = img_bgr[:, mid-2*h:mid+2*h, :]
        return img_crop
    else:
        return img_bgr


def main():
    # img_url = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Image-Rotation/imgs-tmp/20210924/0647430e-bfb4-11eb-9256-0c42a1db69b8-angle-0-20210827.jpg"
    img_url = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Image-Rotation/imgs-tmp/20210924/10a46ade-bfaa-11eb-8b6a-b8599f3b6e92-angle-90-20210827.jpg"
    _, img_bgr = download_url_img(img_url)
    img_bgr = center_crop_by_hw(img_bgr)
    show_img_bgr(img_bgr)


if __name__ == '__main__':
    main()
