# -*- coding: utf-8 -*-
# @Time    : 2021/5/13
# @Author  : Ke
from cnn.Criterions.criterion import criterion
import numpy as np


class L2Loss(criterion):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.type = 'L1Loss'

    def forward(self, input_data, target):
        return np.sum((input_data-target_data)**2)

    def backward(self, input_data, target):
        return 2 * (input_data - target_data)


if __name__ == '__main__':
    l2_loss = L2Loss()
    print(l2_loss)
