# -*- coding: utf-8 -*-
# @Time    : 2021/5/13
# @Author  : Ke
from cnn.Criterions.criterion import criterion
import numpy as np


class cross_entropy(criterion):
    def __init__(self):
        super(cross_entropy, self).__init__()
        self.type = 'cross_entropy'

    def forward(self, input_data, target):
        self.output = np.sum(-target * np.log(input_data), axis=1)
        # self.output = np.mean(self.output)
        self.output = np.sum(self.output)
        return self.output

    def backward(self, input_data, target):
        return input_data - target
