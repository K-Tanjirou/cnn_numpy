# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer


class flatten(layer):
    def __init__(self, **kwargs):
        """
        过渡层
        :param kwargs:
        """
        super(flatten, self).__init__(**kwargs)
        self.type = 'Flatten'

    def forward(self, input_data):
        self.input = input_data
        return input_data.reshape((input_data.shape[0], -1))

    def backward(self, grad_from_back):
        return grad_from_back.reshape(self.input.shape)

    def __repr__(self):
        return f"{self.type}"
