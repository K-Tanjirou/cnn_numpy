# -*- coding: utf-8 -*-
# @Time    : 2021/5/14
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer
from cnn.Utils.utils import *


class conv2D(layer):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, **kwargs):
        """
        2维卷积层
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param kernel: 卷积核大小
        :param stride: 步长
        :param padding: 填充0个数
        """
        super(conv2D, self).__init__(**kwargs)
        self.type = "conv2D"
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = np.zeros((out_channel, in_channel, kernel[0], kernel[1]))
        self.weight_grad = np.zeros((out_channel, in_channel, kernel[0], kernel[1]))
        self.bias = np.zeros((out_channel, 1))
        self.bias_grad = np.zeros((out_channel, 1))

    # 卷积运算, 先用image_to_column将输入数据展开再矩阵相乘即可
    def forward(self, input_data):
        bs, channel, h, w = input_data.shape
        self.input = input_data

        self.output_x = (w - self.kernel[0] + 2 * self.padding[0]) // self.stride[0] + 1
        self.output_y = (h - self.kernel[1] + 2 * self.padding[1]) // self.stride[1] + 1

        self.output = np.zeros((bs, self.out_channel, self.output_y, self.output_x))

        # self.input2col = im2col_v2(input_data, self.kernel, self.stride, self.padding)
        self.input2col = image_to_column(input_data, self.kernel, self.stride, self.padding)

        bs_bais = self.bias.reshape((self.out_channel, 1))
        bs_weight = self.weight.reshape((self.out_channel, -1))         # 让weight展开为n行若干列

        self.output = np.dot(bs_weight, self.input2col) + bs_bais
        self.output = self.output.reshape((self.out_channel, self.output_y, self.output_x) + (bs, ))
        self.output = self.output.transpose(3, 0, 1, 2)

        return self.output

    def backward(self, grad_from_back):
        grad_from_back = grad_from_back.transpose(1, 2, 3, 0).reshape(self.out_channel, -1)
        self.weight_grad = np.dot(grad_from_back, self.input2col.T).reshape(self.weight.shape)          # 该层误差跟输入数据卷积运算
        self.bias_grad = np.sum(grad_from_back, axis=1, keepdims=True)

        # 传导到上一层的误差
        d_col = np.dot(np.reshape(self.weight, (self.out_channel, -1)).T, grad_from_back)
        d_col = d_col.reshape(self.in_channel * np.prod(self.kernel), -1, self.input.shape[0])
        d_col = d_col.transpose(2, 0, 1)
        self.grad_input = col2im(d_col, self.input.shape, self.kernel, self.stride, self.padding)
        return self.grad_input

    def __repr__(self):
        return f"{self.type} with kernel {self.kernel}"
