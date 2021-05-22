# -*- coding: utf-8 -*-
# @Time    : 2021/5/16
# @Author  : Ke
import numpy as np
from tqdm import tqdm
from time import sleep
import cnn.Configurations.configurations as cft
from cnn.Utils.utils import *
from cnn.Layers import *
from cnn.Criterions import *
from cnn.Optimizers import *
from cnn.Models.Model import *
# from cnn.Dataset.data_loader import data_loader
from cnn.Dataset.dataset import *
from cnn.Dataset import *
import matplotlib.pyplot as plt
import time
from numba import jit

# mnist
train_images_path = mnist["train_images_path"]
train_labels_path = mnist["train_labels_path"]
test_images_path = mnist["test_images_path"]
test_labels_path = mnist["test_labels_path"]

# fashion_mnist
# train_images_path = fashion_mnist["train_images_path"]
# train_labels_path = fashion_mnist["train_labels_path"]
# test_images_path = fashion_mnist["test_images_path"]
# test_labels_path = fashion_mnist["test_labels_path"]

# 超参数
epochs = 200
batch_size = 128
normalize = False   # 是否对数据集标准化，归一化将数据值调整到[0, 1]，标准化将归一化的数据调整到[-1, 1]
class_num = 10   # 类别数
loss_all = []
acc_all = []


def data_loader(x, y, bs, e):
    length = len(x)

    for batch in tqdm(range(0, length, bs), desc=f"epoch_{e+1}"):
        batch_x = x[batch:batch + bs]
        batch_y = y[batch:batch + bs]
        yield batch_x, batch_y


def lenet_model():

    conv1 = conv2D(1, 6, [5, 5], [1, 1], [2, 2])
    pooling1 = averagepool([2, 2], [2, 2], [0, 0])

    conv2 = conv2D(6, 16, [5, 5], [1, 1], [0, 0])
    pooling2 = averagepool([2, 2], [2, 2], [0, 0])

    flatten1 = flatten()

    linear1 = linear(400, 120)
    linear2 = linear(120, 84)
    linear3 = linear(84, 10)

    lenet.add(conv1, activation_f="relu")
    lenet.add(pooling1)
    lenet.add(conv2, activation_f="relu")
    lenet.add(pooling2)
    lenet.add(flatten1)
    lenet.add(linear1, activation_f='relu')
    lenet.add(linear2, activation_f='relu')
    lenet.add(linear3, activation_f='softmax')


if __name__ == '__main__':
    # 数据预处理
    train_images = load_train_images(train_images_path)
    train_labels = load_train_labels(train_labels_path)
    test_images = load_test_images(test_images_path)
    test_labels = load_test_labels(test_labels_path)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    train_images = img_reshape(train_images)
    test_images = img_reshape(test_images)

    lenet = model()
    lenet_model()
    lenet.compile(optimizer="Adam", init_parameters="xavier", loss="cross_entropy", rand_seed=cft.seed)
    lenet.summary()
    for epoch in range(epochs):
        train_loader = data_loader(train_images, train_labels, batch_size, epoch)
        start_time = time.time()
        loss, acc = lenet.fit(train_loader)
        end_time = time.time()
        acc_all.append(acc)
        loss_all.append(loss)
        pre_test = lenet.predict_classes(test_images)
        acc_test = accuracy_score(pre_test, test_labels)
        print(f"train_loss={loss}, train_acc={acc}, test_acc={acc_test}, train_time={end_time - start_time}")

    # print("--" * 50)
    # print(f"test accuracy = {acc_test}")
    # print("--" * 50)
    model_plot(loss_all, acc_all, epochs)
