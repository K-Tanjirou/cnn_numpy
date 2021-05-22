# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke
class criterion(object):
    def __init__(self):
        self.type = 'criterion'

    def forward(self, input_data, target):
        raise NotImplementedError

    def backward(self, input_data, target):
        raise NotImplementedError
