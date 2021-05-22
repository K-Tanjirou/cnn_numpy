# -*- coding: utf-8 -*-
# @Time    : 2021/5/21
# @Author  : Ke
import numpy as np


class AdaGrad(object):
    def __init__(self, parameters):
        """
        梯度下降法
        :param parameters: 预先设置的参数
        """
        self.lr = parameters["lr"]

    def __call__(self, layer_stack):
        for layer in layer_stack:
            if layer.type == 'conv2D' or layer.type == 'linear':
                w = layer.get_weights()
                b = layer.get_bias()
                w_grad = layer.get_weight_grad()
                b_grad = layer.get_bias_grad()
                w -= self.lr * w_grad
                b -= self.lr * b_grad
                layer.set_weights(w)
                layer.set_bias(b)
