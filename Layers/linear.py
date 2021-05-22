# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer


class linear(layer):
    def __init__(self, in_channel, out_channel, **kwargs):
        """
        线性层
        :param in_channel: 该层输入结点数
        :param out_channel: 该层输出结点数
        :param type: 定义该层属性为"linear"
        :param weight: 该层权重
        :param bais: 该层偏移量
        """
        super(linear, self).__init__(**kwargs)
        self.type = "linear"
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = np.zeros((in_channel, out_channel))
        self.weight_grad = np.zeros((in_channel, out_channel))
        self.bias = np.zeros((1, out_channel))
        self.bias_grad = np.zeros((1, out_channel))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weight) + self.bias
        return self.output

    def backward(self, grad_from_back):
        bs, _ = self.input.shape

        self.bias_grad = np.sum(grad_from_back, axis=0)
        self.weight_grad = np.sum(grad_from_back.reshape((bs, 1, -1)) * self.input.reshape((bs, -1, 1)), axis=0)

        self.grad = np.dot(grad_from_back, self.weight.T)               # 神经网络与深度学习P94，公式4.63中后面一项
        return self.grad

    def __repr__(self):
        return f"{self.type} layer with {self.out_channel} nodes"


if __name__ == '__main__':
    pass
