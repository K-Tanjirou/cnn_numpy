# -*- coding: utf-8 -*-
# @Time    : 2021/5/13
# @Author  : Ke
from cnn.Criterions.criterion import criterion
import numpy as np


class L1Loss(criterion):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.type = 'L1Loss'

    def forward(self, input_data, target):
        return np.sum(np.abs(input_data, target_data))

    def backward(self, input_data, target):
        s1 = np.greater(input_data, target_data).astype('uint8')
        s_01 = np.greater(target_data, input_data).astype('uint8')

        return s1 - s_01
