# -*- coding: utf-8 -*-
# @Time    : 2021/5/15
# @Author  : Ke
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import sleep
import cnn.Configurations.configurations as cft
from cnn.Utils.utils import *
from cnn.Layers import *
from cnn.Criterions import *
import matplotlib.pyplot as plt
from cnn.Models.Model import model
from cnn.Dataset.data_loader import *

# 训练集文件
train_images_path = r'../Dataset/mnist_dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_path = r'../Dataset/mnist_dataset/train-labels.idx1-ubyte'

# 测试集文件
test_images_path = r'../Dataset/mnist_dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_path = r'../Dataset/mnist_dataset/t10k-labels.idx1-ubyte'
batch_size = 128


if __name__ == '__main__':
    _train_iter = data_loader(train_images_path, train_labels_path, batch_size)
    train_images, train_labels = _train_iter.__next__()

    conv1 = conv2D(1, 6, [5, 5], [1, 1], [2, 2])
    output = conv1.forward(train_images)
    print(output.shape)

    relu1 = activation("relu")
    output = relu1.forward(output)
    print(output.shape)

    pooling1 = averagepool([2, 2], [2, 2], [0, 0])
    output = pooling1.forward(output)
    print(output.shape)

    conv2 = conv2D(6, 16, [5, 5], [1, 1], [0, 0])
    output = conv2.forward(output)
    print(output.shape)

    relu2 = activation("relu")
    output = relu2.forward(output)
    print(output.shape)

    pooling2 = maxpool([2, 2], [2, 2], [0, 0])
    output = pooling2.forward(output)
    print(output.shape)

    Flatten = flatten()
    output = Flatten.forward(output)
    print(output.shape)

    linear1 = linear(400, 120)
    output = linear1.forward(output)
    print(output.shape)

    linear2 = linear(120, 84)
    output = linear2.forward(output)
    print(output.shape)
