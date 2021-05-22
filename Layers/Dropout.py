# -*- coding: utf-8 -*-
# @Time    : 2021/5/21
# @Author  : Ke
import numpy as np
from cnn.Layers.layer import layer
from cnn.Utils.utils import *


class Dropout(Layer):
    def __init__(self, prob, **kwags):
        """
        训练时随机丢弃一些神经元，预测的时候不随机丢弃
        :param prob:
        :param kwags:
        """
        super(Dropout, self).__init__(**kwags)
        self.type = 'dropout'
        self.prob = prob or 0.5
        self.isTrain = True

    def forward(self, input_data):
        self.input = input_data
        if self.isTrain == True:
            self.saved = 1.0 - self.prob
            self.sample = np.random.binomial(n=1, p=self.saved, size=input_data.shape)          # 随机失活
            input_data = input_data * self.sample
            self.output = input_data / self.saved
        else:
            self.output = self.input
        return self.output

    def backward(self, input_data, grad_from_back):

        grad_from_back = grad_from_back.reshape(input_data.shape)

        self.grad_input = grad_from_back / self.saved * self.sample

        return self.grad_input

    def __repr__(self):
        return self.type


    def train(self):
        self.isTrain = True

    def evaluate(self):
        self.isTrain = False
