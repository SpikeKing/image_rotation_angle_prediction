#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 31.8.21
"""

from myutils.cv_utils import *
from myutils.project_utils import *
from x_utils.vpf_sevices import *


def vpf_test():
    # image_url = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Image-Rotation/datasets/img-translation-v1/O1CN01PkiPXO1zZTPC5SkGi_!!6000000006728-0-quark-angle-180-20210824.jpg"
    # image_url = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Image-Rotation/datasets/img-translation-v1/112f1a4a-f046-11eb-aa1f-0c42a168337a-angle-180-20210825.jpg"
    image_url = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/Image-Rotation/datasets/img-translation-v1/4b895e92-bfab-11eb-a77b-0c42a1db69b8-angle-0-20210827.jpg"
    # res_dict = get_vpf_service(img_url=image_url, service_name="ysu362VFeRZkizhfBkfbck")  # 图像翻译
    # res_dict = get_vpf_service(img_url=image_url, service_name="qPEdfEwcvDNKAHpGCLYjBK")  # 自然场景
    res_dict = get_vpf_service(img_url=image_url, service_name="jXLqEieBHp5SvmSmDmAiQS")  # 表格
    print("res_dict: {}".format(res_dict))
    angle = res_dict["data"]["angle"]
    _, img_bgr = download_url_img(image_url)
    img_bgr = rotate_img_for_4angle(img_bgr, angle)
    show_img_bgr(img_bgr)


def main():
    vpf_test()


if __name__ == '__main__':
    main()
