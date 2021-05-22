# -*- coding: utf-8 -*-
# @Time    : 2021/5/10
# @Author  : Ke
import numpy as np
import os
import matplotlib.pyplot as plt

__all__ = ["im2col_v1", "im2col_v2", "col2im", "one_hot", "image_to_column"]


# 卷积乘法中用到的im2col和col2im，参考：https://blog.csdn.net/tanmx219/article/details/82848841
def im2col_v1(img, ksize, stride=1):
    """
    参考： https://github.com/leeroee/NN-by-Numpy/tree/master/package/layers
    但是有更高效的实现方式，参考： https://zhuanlan.zhihu.com/p/64933417
    """
    H, N, W, C = img.shape
    out_h = (H - ksize) // stride + 1
    out_w = (W - ksize) // stride + 1
    col = np.empty((N * out_h * out_w, ksize * ksize * C))
    outsize = out_w * out_h
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + ksize
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + ksize
            col[y_start + x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].reshape(N, -1)
    return col


def im2col_v2(imgray, k, stride, padding):
    """
    im2col讲图像分割成块
    :param imgray: 输入图像大小
    :param k: 卷积核大小
    :param stride: 步长
    :param padding: 填补零个数
    :return: 块矩阵
    """
    bs, ch, h, w = imgray.shape
    r, c = k

    sr, sc = stride

    pr, pc = padding

    output_x = (w + 2 * pr - r) // sr + 1
    output_y = (h + 2 * pc - c) // sc + 1
    # print(output_x)

    # padding
    pd = np.zeros((bs, ch, h + 2 * pr, w + 2 * pc))
    pd[:, :, pr:pr + h, pc:pc + w] = imgray
    # pd = np.pad(imgray, ((pr, pr), (pc, pc)))

    re = np.zeros((bs, ch * r * c, output_x * output_y), dtype='float32')
    count = 0
    for x in range(0, w + 2 * pc - c + 1, sc):
        for y in range(0, h + 2 * pr - r + 1, sr):
            re[:, :, count] = np.reshape(pd[:, :, y:y + r, x:x + c], (bs, -1))
            count = count + 1
    return re


def col2im(d_col, shape_x, k, stride, padding):
    """
    col2im将块还原
    :param d_col: col矩阵
    :param shape_x: 输入数据的维度
    :param k: 卷积核大小
    :param stride: 步长
    :param padding: 填充0个数
    :return:
    """
    bs, ch, h, w = shape_x
    r, c = k

    sr, sc = stride

    pr, pc = padding

    re = np.zeros((bs, ch, h + 2 * pr, w + 2 * pc), dtype='float32')

    count = 0
    for x in range(0, w + 2 * pc - c + 1, sc):
        for y in range(0, h + 2 * pr - r + 1, sr):
            re[:, :, y:y + r, x:x + c] += np.reshape(d_col[:, :, count], (bs, ch, k[0], k[1]))
            count = count + 1

    return re[:, :, pr:pr+h, pc:pc+w]


# label向量化
def one_hot(sequences, dim):
    results = np.zeros((len(sequences), dim))
    for i, sequence in enumerate(sequences):
        results[i, int(sequence)] = 1
    return results


# https://www.bilibili.com/video/BV1Dx411n7UE?p=5
def image_to_column(image, kernel, stride, padding):
    bs, channel, h, w = image.shape

    kr, kc = kernel
    sr, sc = stride
    pr, pc = padding

    output_x = (w + 2 * pr - kr) // sr + 1
    output_y = (h + 2 * pc - kc) // sc + 1

    pd = np.zeros((bs, channel, h + 2 * pr, w + 2 * pc))
    pd[:, :, pr:pr + h, pc:pc + w] = image

    i0 = np.repeat(np.arange(kr), kc)
    i0 = np.tile(i0, channel)       # tile函数将矩阵横向或纵向复制
    i1 = sr * np.repeat(np.arange(output_y), output_x)
    j0 = np.tile(np.arange(kc), kr * channel)
    j1 = sc * np.tile(np.arange(output_x), output_y)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channel), kr * kc).reshape(-1, 1)

    cols = pd[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(kr * kc * channel, -1)
    return cols


if __name__ == "__main__":
    im = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
                   [[[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13], [14, 15, 16, 17]]]])
    kernel = [3, 3]
    stride = [1, 1]
    padding = [0, 0]
    print(im.shape)
    # print(stride_tricks.as_strided(im, shape=kernel, strides=stride))
    out = image_to_column(im[0:1], kernel, stride, padding)
    out = out.T
    print(out.shape)
    print(out)
