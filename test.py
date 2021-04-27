#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 26.11.20
"""

import base64
import logging
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image, ImageSequence, ExifTags
from cairosvg import svg2png
from turbojpeg import TJPF_RGB

from myutils.cv_utils import show_img_bgr

IMAGE_ORIENTATION_TL = 1  # Horizontal (normal)
IMAGE_ORIENTATION_TR = 2  # Mirrored horizontal
IMAGE_ORIENTATION_BR = 3  # Rotate 180
IMAGE_ORIENTATION_BL = 4  # Mirrored vertical
IMAGE_ORIENTATION_LT = 5  # Mirrored horizontal & rotate 270 CW
IMAGE_ORIENTATION_RT = 6  # Rotate 90 CW
IMAGE_ORIENTATION_RB = 7  # Mirrored horizontal & rotate 90 CW
IMAGE_ORIENTATION_LB = 8  # Rotate 270 CW


class DecodeQuarkEducationImage(object):
    def __init__(self):
        self.orientation = None
        for self.orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[self.orientation] == 'Orientation':
                break

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
        nparr = np.asarray(bytearray(image_content)).reshape(1, -1)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        height_, width_ = img.shape[0:2]
        if img.ndim == 2:
            channel_ = 1
        else:
            channel_ = img.shape[-1]
        if channel_ == 4:
            return DecodeQuarkEducationImage._process_rgba(img)
        elif channel_ == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channel_ == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def _decode_jpg(self, image_content):
        try:
            img = self.jpeg.decode(image_content, pixel_format=TJPF_RGB)
        except Exception as e:
            logger.warning('Failed to decode JPEG with TurboJPEG, fallback to OpenCV, error: {}'.format(e))
            nparr = np.asarray(bytearray(image_content)).reshape(1, -1)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        pil_img = Image.open(BytesIO(image_content))
        exif = pil_img._getexif()
        if exif is not None:
            exif_d = dict(exif.items())
            if self.orientation in exif_d:
                orientation = exif_d[self.orientation]
                if orientation == IMAGE_ORIENTATION_TL:
                    # Horizontal (normal)
                    pass
                elif orientation == IMAGE_ORIENTATION_TR:
                    # Mirrored horizontal
                    img = cv2.flip(img, 1, img)
                elif orientation == IMAGE_ORIENTATION_BR:
                    # Rotate 180
                    img = cv2.flip(img, -1, img)
                elif orientation == IMAGE_ORIENTATION_BL:
                    # Mirrored vertical
                    img = cv2.flip(img, 0, img)
                elif orientation == IMAGE_ORIENTATION_LT:
                    # Mirrored horizontal & rotate 270 CW
                    img = cv2.transpose(img, img)
                elif orientation == IMAGE_ORIENTATION_RT:
                    # Rotate 90 CW
                    img = cv2.transpose(img, img)
                    img = cv2.flip(img, 1, img)
                elif orientation == IMAGE_ORIENTATION_RB:
                    # Mirrored horizontal & rotate 90 CW
                    img = cv2.transpose(img, img)
                    img = cv2.flip(img, -1, img)
                elif orientation == IMAGE_ORIENTATION_LB:
                    # Rotate 270 CW
                    img = cv2.transpose(img, img)
                    img = cv2.flip(img, 0, img)
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
            return DecodeQuarkEducationImage._process_rgba(img)
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
            return DecodeQuarkEducationImage._process_rgba(img)
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
            return DecodeQuarkEducationImage._process_rgba(img)
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
        return DecodeQuarkEducationImage._process_rgba(img)

    def do_process(self,
                   image_url=None,
                   image_content=None,
                   image_encode=None):
        logger.info('Image url %s' % image_url)
        if image_encode is not None and len(image_encode) > 0:
            image_content = base64.urlsafe_b64decode(image_encode)
            nparr = np.asarray(bytearray(image_content)).reshape(1, -1)
        elif image_content is not None and len(image_content) > 0:
            pass
        elif image_url is not None:
            with requests.session() as s:
                try:
                    r = s.get(image_url, timeout=20)
                    r.raise_for_status()
                except Exception as e:
                    logger.error('Failed to get from url {}, error: {}'.format(image_url, e))
                    return None, b'', False
                image_content = r.content
        else:
            logger.error('No valid input provided')
            return None, b'', False

        if image_content is None or not len(image_content):
            return None, b'', False
        head = image_content[:10]
        if head[0:4] == b'\xff\xd8\xff\xe0':
            img = self._decode_jpg(image_content)
        elif head[0:6] in (b'GIF87a', b'GIF89a'):
            img = self._decode_gif(image_content)
        elif head[1:4] == b'PNG':
            img = self._decode_png(image_content)
        elif head[4:7] == b'svg':
            img = self._decode_svg(image_content)
        else:
            nparr = np.asarray(bytearray(image_content)).reshape(1, -1)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            if image_url.split('?')[0].endswith('.ico'):
                img = self._decode_ico(image_content)
            elif image_url.split('?')[0].endswith('.svg'):
                img = self._decode_svg(image_content)
            else:
                try:
                    img = self._decode_pil(image_content)
                except Exception as e:
                    logger.error('{}'.format(e))
                    pass

        if img is None:
            return None, b'', False

        img = img.astype(np.uint8)

        h, w, c = img.shape
        logger.info("success: {}, {}, {}".format(h, w, c))
        return img, image_content, True

def main():
    # img_path = os.path.join(DATA_DIR, 'cases', '00002_ans_img.jpg')
    img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_rotation/datasets/zjw_imgs_20210427/00002_ans_img.jpg"
    dd = DecodeQuarkEducationImage()
    img, image_content, x = dd.do_process(img_url)

    # cv2.imwrite(out_path, im)
    # show_img_bgr(im)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.uint8)
    show_img_bgr(img)

    # angle = 90
    # new_im1 = rotate_img_for_4angle(im, (360 - angle) % 360)
    cv2.imwrite("xxxx.jpg", img)
    #
    # # new_im2 = rotate_img_with_bound(im, angle)  # 第2种旋转模型
    #
    # new_im2, angle, rhw_ratio = generate_rotated_image(new_im1, 10, crop_largest_rect=True)
    # cv2.imwrite(out2_path, new_im2)

    # show_img_bgr(new_im1)
    # show_img_bgr(new_im2)
    # show_img_bgr(new_im3)


if __name__ == '__main__':
    main()
