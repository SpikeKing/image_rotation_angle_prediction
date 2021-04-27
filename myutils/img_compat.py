#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 27.4.21
"""

import base64
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image, ImageSequence
from cairosvg import svg2png


class ImgCompatBGR(object):
    def __init__(self):
        pass

    @staticmethod
    def _process_rgba(img):
        alpha_channel = img[:, :, 3]
        h, w = alpha_channel.shape
        alpha_mask = alpha_channel == 255
        if np.amax(alpha_channel) != 0 and np.sum(alpha_mask) != (h * w):
            img = np.empty(alpha_channel.shape, np.uint8)
            img.fill(255)
            img -= alpha_channel
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    @staticmethod
    def _decode_png(image_content):
        np_arr = np.asarray(bytearray(image_content)).reshape(1, -1)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            channel_ = 1
        else:
            channel_ = img.shape[-1]
        if channel_ == 4:
            return ImgCompatBGR._process_rgba(img)
        elif channel_ == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channel_ == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _decode_jpg(image_content):
        np_arr = np.asarray(bytearray(image_content)).reshape(1, -1)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _decode_svg(image_content):
        png = svg2png(bytestring=image_content)
        pil_img = Image.open(BytesIO(png)).convert('RGBA')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        if img.ndim == 2:
            channel_ = 1
        else:
            channel_ = img.shape[-1]
        if channel_ == 4:
            return ImgCompatBGR._process_rgba(img)
        elif channel_ == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channel_ == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _decode_ico(image_content):
        pil_img = Image.open(BytesIO(image_content)).convert('RGBA')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        if img.ndim == 2:
            channel_ = 1
        else:
            channel_ = img.shape[-1]
        if channel_ == 4:
            return ImgCompatBGR._process_rgba(img)
        elif channel_ == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channel_ == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _decode_pil(image_content):
        pil_img = Image.open(BytesIO(image_content)).convert('RGBA')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        if img.ndim == 2:
            channel_ = 1
        else:
            channel_ = img.shape[-1]
        if channel_ == 4:
            return ImgCompatBGR._process_rgba(img)
        elif channel_ == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channel_ == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _decode_gif(image_content):
        f = BytesIO(image_content)
        im = Image.open(f)

        img = None
        for frame in ImageSequence.Iterator(im):
            img = np.asarray(frame.convert('RGBA'))
        return ImgCompatBGR._process_rgba(img)

    @staticmethod
    def imread(image_path=None, image_url=None, image_content=None, image_encode=None):
        """
        读取图像
        """
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                return img_bgr
        except Exception as e:
            pass

        if image_path:
            image = Image.open(image_path)
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format=image.format)
            image_content = img_byte_arr.getvalue()
        elif image_encode is not None and len(image_encode) > 0:
            image_content = base64.urlsafe_b64decode(image_encode)
        elif image_content is not None and len(image_content) > 0:
            pass
        elif image_url is not None:
            with requests.session() as s:
                try:
                    r = s.get(image_url, timeout=20)
                    r.raise_for_status()
                except Exception as e:
                    return None
                image_content = r.content
        else:
            return None

        if image_content is None or not len(image_content):
            return None

        head = image_content[:10]
        if head[0:4] == b'\xff\xd8\xff\xe0':
            img = ImgCompatBGR._decode_jpg(image_content)
        elif head[0:6] in (b'GIF87a', b'GIF89a'):
            img = ImgCompatBGR._decode_gif(image_content)
        elif head[1:4] == b'PNG':
            img = ImgCompatBGR._decode_png(image_content)
        elif head[4:7] == b'svg':
            img = ImgCompatBGR._decode_svg(image_content)
        else:
            np_arr = np.asarray(bytearray(image_content)).reshape(1, -1)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            if image_url.split('?')[0].endswith('.ico'):
                img = ImgCompatBGR._decode_ico(image_content)
            elif image_url.split('?')[0].endswith('.svg'):
                img = ImgCompatBGR._decode_svg(image_content)
            else:
                try:
                    img = ImgCompatBGR._decode_pil(image_content)
                except Exception as e:
                    pass

        if img is None:
            return None

        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


def test_url():
    img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_rotation/datasets/zjw_imgs_20210427/00002_ans_img.jpg"
    from myutils.project_utils import download_url_img
    is_ok, parsed_image = download_url_img(img_url)
    print('[Info] parsed_image is None: {}'.format(parsed_image == None))
    img = ImgCompatBGR.imread(image_url=img_url)
    from myutils.cv_utils import show_img_bgr
    show_img_bgr(img)


def image_to_byte_array(image):
    imgByteArr = BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def test_path():
    import os
    from root_dir import DATA_DIR
    img_path = os.path.join(DATA_DIR, "cases", "00002_ans_img.jpg")
    parsed_image = cv2.imread(img_path)
    print('[Info] parsed_image is None: {}'.format(parsed_image == None))
    img = ImgCompatBGR.imread(image_path=img_path)
    from myutils.cv_utils import show_img_bgr
    show_img_bgr(img)


def main():
    # test_url()
    test_path()


if __name__ == '__main__':
    main()
