# -*- coding: utf-8 -*-
# @Time    : 2021/5/14
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer
from cnn.Utils.utils import *


class averagepool(layer):
    def __init__(self, kernel, stride, padding):
        """
        平均池化层
        :param kernel: 卷积核大小
        :param stride: 步长
        :param padding: 填充0个数
        """
        super(averagepool, self).__init__()
        self.type = 'averagepool'
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, input_data):
        self.input = input_data
        bs, channel, h, w = input_data.shape

        self.output_x = (w - self.kernel[0] + 2 * self.padding[0]) // self.stride[0] + 1
        self.output_y = (h - self.kernel[1] + 2 * self.padding[1]) // self.stride[1] + 1

        input_data = input_data.reshape(bs * channel, 1, h, w)
        self.input2col = image_to_column(input_data, self.kernel, self.stride, self.padding)

        self.output = np.mean(self.input2col, axis=0)           # 沿着列水平计算均值
        self.output = self.output.reshape(self.output_y, self.output_x, bs, channel)
        self.output = self.output.transpose(2, 3, 0, 1)
        # print(self.output.shape)
        return self.output

    def backward(self, grad_from_back):
        bs, oc, h, w = grad_from_back.shape

        grad_from_back = grad_from_back.transpose(2, 3, 0, 1).ravel()
        grad_col = np.zeros((np.prod(self.kernel), grad_from_back.size))
        grad_col[:, range(grad_from_back.size)] = 1. / grad_col.shape[0] * grad_from_back

        grad_col = grad_col.reshape(bs, oc * self.kernel[0] * self.kernel[1], h * w)
        self.grad_input = col2im(grad_col, self.input.shape, self.kernel, self.stride, self.padding)

        return self.grad_input

    def __repr__(self):
        return f"{self.type}"
