# -*- coding: utf-8 -*-
# @Time    : 2021/5/12
# @Author  : Ke

# 常用的激活函数有sigmoid, relu, tanh, softplus, PReLU, ELU, Swish, mish
activation_function = ("sigmoid", "relu", "softmax", "PReLU", "Swish")
init_parameters = ("normal", "xavier", "kaiming", "myself")
optimizer = ("SGD", "Adam", "RMSprop")

# Adam优化器参数
Adam_parameters = {
    "lr": 0.0001,
    "beta1": 0.9,
    "beta2": 0.999,
}

# SGD优化器参数
SGD_parameters = {
    "lr": 0.0001,
}

seed = 666
