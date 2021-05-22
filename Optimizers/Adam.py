# -*- coding: utf-8 -*-
# @Time    : 2021/5/16
# @Author  : Ke
import numpy as np


class Adam(object):
    def __init__(self, parameters):
        """
        Adam优化算法
        :param parameters: 预先设置的参数
        """
        self.lr = parameters["lr"]
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        self.eps = 1e-8
        self.t = 0

    def __call__(self, layer_stack):
        for layer in layer_stack:
            if layer.type == 'conv2D' or layer.type == 'linear':
                layer.mw = np.zeros(layer.get_weight_grad().shape)
                layer.vw = np.zeros(layer.get_weight_grad().shape)

                layer.mb = np.zeros(layer.get_bias_grad().shape)
                layer.vb = np.zeros(layer.get_bias_grad().shape)

                self.t += 1

                w = layer.get_weights()
                b = layer.get_bias()

                layer.mw = layer.mw * self.beta1 + (1 - self.beta1) * layer.get_weight_grad()
                layer.vw = layer.vw * self.beta2 + (1 - self.beta2) * layer.get_weight_grad() * layer.get_weight_grad()
                layer.denomw = np.sqrt(layer.vw) + self.eps

                layer.mb = layer.mb * self.beta1 + (1 - self.beta1) * layer.get_bias_grad()
                layer.vb = layer.vb * self.beta2 + (1 - self.beta2) * layer.get_bias_grad() * layer.get_bias_grad()
                layer.denomb = np.sqrt(layer.vb) + self.eps

                bc1 = 1 - self.beta1 ** self.t
                bc2 = 1 - self.beta2 ** self.t

                ss = - self.lr * np.sqrt(bc2) / bc1

                w = w + layer.mw / layer.denomw * ss
                b = b + layer.mb / layer.denomb * ss

                layer.set_weights(w)
                layer.set_bias(b)

