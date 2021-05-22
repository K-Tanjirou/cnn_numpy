# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


a = 0.5     # PReLU参数, 定义域[0, 1]
beta = 10   # Swish参数，无限制


class activation(layer):
    def __init__(self, act):
        """
        激活函数
        :param act: 可选有"relu", "sigmoid", "PReLU", "Swish"
        """
        super(activation, self).__init__()
        self.type = "activation"
        self.act = act

    def forward(self, input_data):
        self.input = input_data
        if self.act == 'relu':
            self.output = np.maximum(input_data, np.zeros(input_data.shape))
        if self.act == 'sigmoid':
            self.output = sigmoid(input_data)
        if self.act == 'PReLU':
            self.output = np.maximum(input_data, a * input_data)
        if self.act == 'Swish':
            self.output = input_data * sigmoid(beta * input_data)

        return self.output

    def backward(self, grad_from_back):
        if self.act == 'relu':
            self.grad_input = grad_from_back * np.greater(self.input, 0)
        if self.act == 'sigmoid':
            self.grad_input = grad_from_back * (sigmoid(self.input) * (1 - sigmoid(self.input)))
        if self.act == 'PReLU':
            self.grad_input = grad_from_back * self.output / self.input
        if self.act == 'Swish':
            self.output = sigmoid(beta * self.input) + self.input * beta * sigmoid(beta * self.input) * (1 - sigmoid(beta * self.input))

        return self.grad_input

    def __repr__(self):
        return f"{self.act} activation function"
