# -*- coding: utf-8 -*-
# @Time    : 2021/5/14
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer
from cnn.Utils.utils import *


class maxpool(layer):
    def __init__(self, kernel, stride, padding):
        """
        最大池化层
        :param kernel: 卷积核大小
        :param stride: 步长
        :param padding: 填充0个数
        """
        super(maxpool, self).__init__()
        self.type = 'maxpool'
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, input_data):
        bs, channel, h, w = input_data.shape

        self.input = input_data
        self.output_channel = channel

        self.output_x = (w - self.kernel[0] + 2 * self.padding[0]) // self.stride[0] + 1
        self.output_y = (w - self.kernel[1] + 2 * self.padding[1]) // self.stride[1] + 1

        self.output = np.zeros((bs, self.output_channel, self.output_y, self.output_x))

        self.input2col = im2col_v2(input_data, self.kernel, self.stride, self.padding)
        imcol_all = self.input2col.reshape((bs, channel, self.kernel[0] * self.kernel[1], self.output_x * self.output_y))
        self.output = np.max(imcol_all, 2)
        self.max_idx = np.argmax(imcol_all, 2)

        self.output = self.output.reshape((bs, self.output_channel, self.output_y, self.output_x))
        return self.output

    def backward(self, grad_from_back):
        bs, oc, h, w = grad_from_back.shape
        # (bs,self.input_channel*self.kernel[0]*self.kernel[1],oh*ow)
        # bs ic h*w
        d_col = np.zeros((bs, ic * self.kernel[0] * self.kernel[1], h * w))
        d_col[self.max_idx] = grad_from_back
        self.grad_input = col2im(d_col, self.input.shape, self.kernel, self.stride, self.padding)
        return self.grad_input
