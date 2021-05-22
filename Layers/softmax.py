# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke
import numpy as np
from cnn.Layers import *


class softmax(layer):
    def __init__(self, act):
        """
        softmax层，不参与反向传播
        :param act:
        """
        super(softmax, self).__init__()
        self.type = 'activation'
        self.act = act
        self.epsilon = 1e-12        # 防止求导后分母为0

    def forward(self, input_data):
        # self.input = input_data
        self.output = np.exp(input_data) / (np.sum(np.exp(input_data), axis=1).reshape(-1, 1) + self.epsilon)
        return self.output

    def backward(self, grad_from_back):
        pass

    def __repr__(self):
        return f"{self.act}"
