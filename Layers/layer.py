# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke
import numpy as np


class layer(object):
    count = 0

    def __init__(self, **kwargs):
        """
        父类，所有的层都继承该类的方法和属性
        :param kwargs:
        """
        self.type = 'layer'
        layer.count += 1        # 当实例化一个层后计数加一

    def __call__(self, data):
        self.input = data
        return self.forward(data)

    def forward(self, input_data):
        """
        前向传播
        :param input_data: 输入数据
        :return:
        """
        raise NotImplementedError       # 子类没有实现父类要求一定要实现的接口就报错

    def backward(self, grad_from_back):
        """
        反向传播
        :param grad_from_back: 上一层返回的梯度
        :return:
        """
        raise NotImplementedError

    def get_name(self):
        """
        获取该层的类型
        :return: type
        """
        return self.type

    def get_weights(self):
        """
        获取该层权重参数
        :return:
        """
        return self.weight

    def get_bias(self):
        """
        获取该层偏移量参数
        :return:
        """
        return self.bias

    def get_weight_grad(self):
        """
        获取该层权重梯度
        :return:
        """
        return self.weight_grad

    def get_bias_grad(self):
        """
        获取该层偏移量梯度
        :return:
        """
        return self.bias_grad

    def set_weights(self, weight):
        """
        设置该层的权重参数
        :param weight: 权重参数
        :return:
        """
        self.weight = weight

    def set_bias(self, bias):
        """
        设置该层偏移量参数
        :param bias: 偏移量参数
        :return:
        """
        self.bias = bias


if __name__ == '__main__':
    l1 = layer()
    print(l1.type)
