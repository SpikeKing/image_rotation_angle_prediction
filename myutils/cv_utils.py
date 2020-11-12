#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/3/13
"""


def draw_line_len(img_bgr, start_p, v_length, v_arrow, is_new=True, is_show=False, save_name=None):
    """
    绘制直线
    """
    import cv2
    import copy
    import numpy as np

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    x2 = int(start_p[0] - v_length * np.cos(v_arrow / 360 * 2 * np.pi))
    y2 = int(start_p[1] - v_length * np.sin(v_arrow / 360 * 2 * np.pi))

    cv2.arrowedLine(img_bgr, tuple(start_p), (x2, y2), color=(0, 0, 255), thickness=4, tipLength=0.4)

    if is_show:
        show_img_bgr(img_bgr, save_name=save_name)  # 显示眼睛


def draw_text(img_bgr, text, org=(3, 20), color=(0, 0, 255)):
    import cv2
    h, w, _ = img_bgr.shape
    m = max(h, w)
    text = str(text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = m / float(1000)
    thickness = m // 200
    lineType = 2

    img_bgr = cv2.putText(img_bgr, text, org, font,
                          fontScale, color, thickness, lineType)
    return img_bgr


def draw_eyes(img_bgr, eyes_landmarks, radius, offsets_list, is_new=True, is_show=False, save_name=None):
    """
    绘制图像
    """
    import cv2
    import copy
    import numpy as np

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    th = 1
    eye_upscale = 1
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=eye_upscale, fy=eye_upscale)

    for el, r, offsets in zip(eyes_landmarks, radius, offsets_list):
        start_x, start_y, offset_scale = offsets
        real_el = el * offset_scale + [start_x, start_y]
        real_radius = r * offset_scale

        # 眼睛
        cv2.polylines(
            img_bgr,
            [np.round(eye_upscale * real_el[0:8]).astype(np.int32)
                 .reshape(-1, 1, 2)],
            isClosed=True, color=(255, 255, 0),
            thickness=th, lineType=cv2.LINE_AA,
        )

        # 眼球
        cv2.polylines(
            img_bgr,
            [np.round(eye_upscale * real_el[8:16]).astype(np.int32)
                 .reshape(-1, 1, 2)],
            isClosed=True, color=(0, 255, 255),
            thickness=th, lineType=cv2.LINE_AA,
        )

        iris_center = real_el[16]
        eye_center = real_el[17]

        eye_center = (real_el[0] + real_el[4]) / 2

        # 虹膜中心
        cv2.drawMarker(
            img_bgr,
            tuple(np.round(eye_upscale * iris_center).astype(np.int32)),
            color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
            thickness=th + 1, line_type=cv2.LINE_AA,
        )

        # 眼睑中心
        cv2.drawMarker(
            img_bgr,
            tuple(np.round(eye_upscale * eye_center).astype(np.int32)),
            color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
            thickness=th + 1, line_type=cv2.LINE_AA,
        )

        cv2.circle(img_bgr, center=tuple(eye_center), radius=real_radius, color=(0, 0, 255))

        if is_show:
            show_img_bgr(img_bgr, save_name=save_name)  # 显示眼睛

    return img_bgr


def draw_box(img_bgr, box, color=(0, 0, 255), is_show=True, is_new=True):
    """
    绘制box
    """
    import cv2
    import copy
    import matplotlib.pyplot as plt

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    # print(x_min, y_min, x_max, y_max)

    ih, iw, _ = img_bgr.shape
    # color = (0, 0, 255)
    tk = max(min(ih, iw) // 200, 2)

    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), color, tk)

    if is_show:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()

    return img_bgr


def draw_points(img_bgr, points, is_new=True, save_name=None):
    """
    绘制多个点
    """
    import cv2
    import copy
    import matplotlib.pyplot as plt

    if is_new:
        img_bgr = copy.deepcopy(img_bgr)

    color = (0, 255, 0)
    ih, iw, _ = img_bgr.shape
    r = max(min(ih, iw) // 200, 1)
    tk = -1
    for p in points:
        p = (int(p[0]), int(p[1]))
        cv2.circle(img_bgr, tuple(p), r, color, tk)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_rgb)


def draw_pie(labels, sizes):
    """
    绘制饼状图, 测试
    labels = [u'大型', u'中型', u'小型', u'微型']  # 定义标签
    sizes = [46, 253, 321, 66]  # 每块值

    :param labels: 标签
    :param sizes: 类别值
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.figure(figsize=(9, 9))  # 调节图形大小

    colors = ['yellow', 'darkorange', 'limegreen', 'lightskyblue', 'blueviolet']  # 每块颜色定义
    colors = colors[:len(labels)]
    explode = tuple([0] * len(labels))  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离

    # 设置饼图文字大小
    [t.set_size(20) for t in text1]
    [t.set_size(20) for t in text2]

    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部的文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    plt.show()


def point2box(point, radius):
    """
    点到矩形
    :param point: 点
    :param radius: 半径
    :return: [x_min, y_min, x_max, y_max]
    """
    start_p = [point[0] - radius, point[1] - radius]
    end_p = [point[0] + radius, point[1] + radius]

    return [int(start_p[0]), int(start_p[1]), int(end_p[0]), int(end_p[1])]


def get_mask_box(mask):
    """
    mask的边框
    """
    import numpy as np
    y, x = np.where(mask)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    box = [x_min, y_min, x_max, y_max]
    return box


def get_box_size(box):
    """
    矩形尺寸
    """
    x_min, y_min, x_max, y_max = [b for b in box]
    return (x_max - x_min) * (y_max - y_min)


def get_polygon_size(box):
    """
    四边形尺寸
    """
    import cv2
    import numpy as np
    contour = np.array(box, dtype=np.int32)
    area = cv2.contourArea(contour)
    return area


def get_patch(img, box):
    """
    获取Img的Patch
    :param img: 图像
    :param box: [x_min, y_min, x_max, y_max]
    :return 图像块
    """
    h, w, _ = img.shape
    x_min = int(max(0, box[0]))
    y_min = int(max(0, box[1]))
    x_max = int(min(box[2], w))
    y_max = int(min(box[3], h))

    img_patch = img[y_min:y_max, x_min:x_max, :]
    return img_patch


def expand_patch(img, box, x):
    """
    box扩充x像素
    """
    h, w, _ = img.shape
    x_min = int(max(0, box[0] - x))
    y_min = int(max(0, box[1] - x))
    x_max = int(min(box[2] + x, w))
    y_max = int(min(box[3] + x, h))

    img_patch = img[y_min:y_max, x_min:x_max, :]
    return img_patch


def expand_box(img, box, x):
    """
    box扩充x像素
    """
    h, w, _ = img.shape
    x_min = int(max(0, box[0] - x))
    y_min = int(max(0, box[1] - x))
    x_max = int(min(box[2] + x, w))
    y_max = int(min(box[3] + x, h))

    return [x_min, y_min, x_max, y_max]


def merge_two_box(box_a, box_b):
    """
    合并两个box
    """
    x1_min, y1_min, x1_max, y1_max = box_a
    x2_min, y2_min, x2_max, y2_max = box_b
    nx_min, ny_min = min(x1_min, x2_min), min(y1_min, y2_min)
    nx_max, ny_max = max(x1_max, x2_max), max(y1_max, y2_max)
    tmp_box = [nx_min, ny_min, nx_max, ny_max]

    return tmp_box


def min_iou(box_a, box_b):
    """
    最小框的面积占比
    """
    box_a = [int(x) for x in box_a]
    box_b = [int(x) for x in box_b]

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / min(box_a_area, box_b_area)

    return iou


def mid_point(p1, p2):
    """
    计算中点
    """
    x = (p1[0] + p2[0]) // 2
    y = (p1[1] + p2[1]) // 2
    return [x, y]


def generate_colors(n_colors):
    """
    随机生成颜色
    """
    import numpy as np

    np.random.seed(37)
    color_list = []
    for i in range(n_colors):
        color = (np.random.random((1, 3)) * 0.8 + 0.2).tolist()[0]
        color = [int(j * 255) for j in color]
        color_list.append(color)

    return color_list


def show_img_bgr(img_bgr, save_name=None):
    """
    展示BGR彩色图
    """
    import cv2
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_rgb)


def show_img_gray(img_gray, save_name=None):
    """
    展示灰度图
    """
    import matplotlib.pyplot as plt

    plt.imshow(img_gray)
    plt.show()
    if save_name:
        print('[Info] 存储图像: {}'.format(save_name))
        plt.imsave(save_name, img_gray)


def init_vid(vid_path):
    """
    初始化视频
    """
    import cv2

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 26

    return cap, n_frame, fps, h, w


def unify_size(h, w, ms):
    """
    统一最长边的尺寸

    :h 高
    :w 宽
    :ms 最长尺寸
    """
    # 最长边修改为标准尺寸
    if w > h:
        r = ms / w
    else:
        r = ms / h
    h = int(h * r)
    w = int(w * r)

    return h, w


def get_fixes_frames(n_frame, max_gap):
    """
    等比例抽帧

    :param n_frame: 总帧数
    :param max_gap: 抽帧数量
    :return: 帧索引
    """
    from math import floor

    idx_list = []
    if n_frame > max_gap:
        v_gap = float(n_frame) / float(max_gap)  # 只使用100帧
        for gap_idx in range(max_gap):
            idx = int(floor(gap_idx * v_gap))
            idx_list.append(idx)
    else:
        for gap_idx in range(n_frame):
            idx_list.append(gap_idx)
    return idx_list


def sigmoid_thr(val, thr, gap, reverse=False):
    """
    数值归一化

    thr: 均值
    gap: 区间，4~5等分
    """
    import numpy as np
    x = val - thr
    if reverse:
        x *= -1
    x = x / gap
    sig = 1 / (1 + np.exp(x * -1))
    return round(sig, 4)  # 保留4位


def write_video(vid_path, frames, fps, h, w):
    """
    写入视频
    :param vid_path: 输入视频的URL
    :param frames: 帧列表
    :param fps: FPS
    :param w: 视频宽
    :param h: 视频高
    :return: 写入完成的视频路径
    """
    import cv2
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以
    vw = cv2.VideoWriter(filename=vid_path, fourcc=fourcc, fps=fps, frameSize=(w, h), isColor=True)

    for frame in frames:
        vw.write(frame)

    vw.release()
    return vid_path


def merge_imgs(imgs, cols=6, rows=6, is_h=True):
    """
    合并图像
    :param imgs: 图像序列
    :param cols: 行数
    :param rows: 列数
    :param is_h: 是否水平排列
    :param sk: 间隔，当sk=2时，即0, 2, 4, 6
    :return: 大图
    """
    import numpy as np

    if not imgs:
        raise Exception('[Exception] 合并图像的输入为空!')

    img_shape = imgs[0].shape
    h, w, _ = img_shape

    large_imgs = np.ones((rows * h, cols * w, 3)) * 255  # 大图

    if is_h:
        for j in range(rows):
            for i in range(cols):
                idx = j * cols + i
                if idx > len(imgs) - 1:  # 少于帧数，输出透明帧
                    break
                # print('[Info] 帧的idx: {}, i: {}, j:{}'.format(idx, i, j))
                large_imgs[(j * h):(j * h + h), (i * w): (i * w + w)] = imgs[idx]
                # print(large_imgs.shape)
                # show_png(large_imgs)
        # show_png(large_imgs)
    else:
        for i in range(cols):
            for j in range(rows):
                idx = i * cols + j
                if idx > len(imgs) - 1:  # 少于帧数，输出透明帧
                    break
                large_imgs[(j * h):(j * h + h), (i * w): (i * w + w)] = imgs[idx]

    return large_imgs


def merge_two_imgs(img1, img2):
    """
    左右合并2张图像, 高度相同, 宽度等比例变化
    """
    import cv2
    import numpy as np

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    h = min(h1, h2)
    n_w1 = int(w1 * h / h1)
    n_w2 = int(w2 * h / h2)
    n_img1 = cv2.resize(img1, (n_w1, h))
    n_img2 = cv2.resize(img2, (n_w2, h))

    large_img = np.ones((h, n_w1 + n_w2, 3)) * 255
    large_img[:, 0: n_w1] = n_img1
    large_img[:, n_w1: n_w1+n_w2] = n_img2
    large_img = large_img.astype(np.uint8)

    return large_img


def rotate_img_with_bound(img_np, angle):
    """
    旋转图像角度
    注意angle是顺时针还是逆时针
    """
    import cv2
    import numpy as np

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img_np.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # !!!注意angle是顺时针还是逆时针
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(img_np, M, (nW, nH))


def resize_img_fixed(img, x, is_height=True):
    """
    resize图像，根据某一个边的长度
    """
    import cv2

    h, w, _ = img.shape
    if is_height:
        nh = x
        nw = int(w * nh / h)
    else:
        nw = x
        nh = int(h * nw / w)

    img_r = cv2.resize(img, (nw, nh))
    return img_r

def main():
    labels = [u'大型', u'中型', u'小型', u'微型']  # 定义标签
    sizes = [46, 253, 321, 66]  # 每块值
    labels = [u"多题", u"单题", u"图题", u"异常", u"科目错误"]
    sizes = [26, 8, 5, 2, 9]
    draw_pie(labels, sizes)


if __name__ == '__main__':
    main()