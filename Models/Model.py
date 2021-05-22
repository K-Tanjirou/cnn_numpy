# -*- coding: utf-8 -*-
# @Time    : 2021/5/13
# @Author  : Ke
import numpy as np
import time
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import sleep
import cnn.Configurations.configurations as cft
from cnn.Utils.utils import *
from cnn.Layers import *
from cnn.Criterions import *
from cnn.Optimizers import *
import matplotlib.pyplot as plt
from numba import jit

__all__ = ["accuracy_score", "model_plot", "model"]

# 超参数
epochs = 2000
batch_size = 50
class_num = 3
loss_all = []
acc_all = []


def accuracy_score(output, target):
    return np.mean([1 if int(output[i]) == np.argmax(target[i]) else 0 for i in range(len(output))])


def model_plot(loss_ls, acc_ls, epochs):
    """
    绘图
    :param loss_ls: 损失函数
    :param acc_ls: 精度
    :param epochs: 迭代次数
    :return: None
    """
    plt.plot(range(epochs), loss_ls, label="loss")
    plt.plot(range(epochs), acc_ls, label="acc")
    plt.legend()        # 显示上面的label
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


class model(object):
    def __init__(self):
        self.layer_stack = []
        self.train_flag = True

    def forward(self, input_data):
        """
        前向传播
        :param input_data: 输入数据
        :return: 返回最后一层的结果
        """
        for ly in self.layer_stack:
            input_data = ly.forward(input_data)
        return input_data

    def backward(self, grad_from_output):
        """
        反向传播
        :param grad_from_output: softmax + loss 返回的梯度
        :return: None
        """
        layer_back = self.layer_stack[::-1]
        layer_back = layer_back[1:]
        # print(layer_back)
        for ly in layer_back:
            grad_from_output = ly.backward(grad_from_output)

    def _optimizer(self):
        """
        更新参数
        :return:
        """
        if self.optimizer == "SGD":
            opt = SGD(cft.SGD_parameters)
        elif self.optimizer == "Adam":
            opt = Adam(cft.Adam_parameters)
        elif self.optimizer == "RMSprop":
            opt = RMS
        opt(self.layer_stack)

    def add(self, i_layer, activation_f=None):
        """
        仿造keras向模型添加隐藏层，如果要仿造pytorch，可以尝试gc.get_objects()方法
        :param i_layer: 添加layers中的层，有全连接层，卷积层，池化层
        :param activation_f: 各层的激活函数
        :return: None
        """
        self.layer_stack.append(i_layer)
        if activation_f is not None:
            if activation_f != "softmax":
                self.layer_stack.append(activation(activation_f))
            else:
                self.layer_stack.append(softmax(activation_f))

    def compile(self, optimizer, init_parameters, loss, rand_seed=None):
        """
        设置模型的优化方法，初始化方法以及准则
        :param optimizer: 优化方法
        :param init_parameters: 初始化方法
        :param loss: 损失函数
        :return:
        """
        if init_parameters not in cft.init_parameters:
            raise ValueError("没有此初始化方法")
        if optimizer not in cft.optimizer:
            raise ValueError("没有此优化方法")
        if rand_seed is not None:
            np.random.seed(rand_seed)

        self.init_parameters = init_parameters
        self._init_parameter()
        self.optimizer = optimizer
        if loss == "cross_entropy":
            self.loss_function = cross_entropy()
        elif loss == "l1_loss":
            self.loss_function = L1Loss()
        elif loss == "l2_loss":
            self.loss_function = L2Loss()

    def _init_parameter(self):
        """
        各层参数初始化，可选方法有"normal", "xavier", "kaiming", 常用的初始化方法还有MSRA, He，后续可继续实现
        normal: 根据高斯分布做初始化
        xavier: Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks[J]. Journal of Machine Learning Research, 2010, 9:249-256.
        kaiming: He K, Zhang X, Ren S, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1026-1034.
        :return: None
        """
        init_type = self.init_parameters

        for i_layer in self.layer_stack:
            if i_layer.get_name() == 'conv2D' or i_layer.get_name() == 'linear':
                if init_type == 'normal':
                    i_layer.set_weights(np.random.normal(0, 0.02, i_layer.get_weights().shape))
                    i_layer.set_bias(np.random.normal(0, 0.02, i_layer.get_bias().shape))

                if init_type == 'myself':
                    i_layer.set_weights(np.random.randn(i_layer.in_channel, i_layer.out_channel) * 0.01)
                    i_layer.set_bias(np.random.randn(1, i_layer.out_channel) * 0.1)

                elif init_type == 'xavier':
                    i_layer.set_weights(np.random.uniform(-np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                          np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                          i_layer.get_weights().shape))
                    i_layer.set_bias(np.random.uniform(-np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                       np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                       i_layer.get_bias().shape))

                elif init_type == 'kaiming':
                    i_layer.set_weights(np.random.uniform(-np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                          np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                          i_layer.get_weights().shape))
                    i_layer.set_bias(np.random.uniform(-np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                       np.sqrt(6 / (i_layer.in_channel + i_layer.out_channel)),
                                                       i_layer.get_bias().shape))

    def train(self):
        """
        设置网络为训练模式
        :return:
        """
        self.train_flag = True
        if not self.layer_stack:
            return
        for ly in self.layer_stack:
            if ly.type == 'dropout':
                ly.train()

    def evaluate(self):
        """
        设置网络为预测模式
        :return:
        """
        self.train_flag = False
        if not self.layer_stack:
            return
        for ly in self.layer_stack:
            if ly.type == 'dropout':
                ly.evaluate()

    def get_Status(self):
        return self.train_flag

    def fit(self, dl):
        """
        将数据输入到模型中并训练
        :return:
        """
        loss = 0
        num = 0
        accu = 0
        for x, y in dl:
            # sleep(0.000005)
            self.grad_stack = []
            output = self.forward(x)
            pre_train = np.argmax(output, axis=1)
            accu += np.sum([1 if int(pre_train[i]) == np.argmax(y[i]) else 0 for i in range(len(x))])
            loss += self.loss_function.forward(output, y)
            d_loss = self.loss_function.backward(output, y)
            self.grad_stack.append(d_loss)
            self.backward(d_loss)
            self._optimizer()
            num += len(x)
        return loss / num, accu / num

    def predict(self, input_data):
        """
        预测概率输出
        :param input_data: 输入数据
        :return: 预测
        """
        output = self.forward(input_data)
        return output

    def predict_classes(self, input_data):
        """
        预测样本属于哪个类
        :param input_data: 输入数据
        :return: 类
        """
        pred = self.predict(input_data)
        pre_class = np.argmax(pred, axis=1)
        return pre_class

    def summary(self):
        """
        输出模型结构
        :return: None
        """
        layer_num = layer.count
        print("-"*20 + "Network Architecture" + "-"*20)
        for l in range(layer_num):
            print(f"{self.layer_stack[l]}")
            print("*"*30)
        print('-'*60)

    def save(self, file):
        """
        保存模型, 当file不存在时自动创建
        :param file:
        :return:
        """
        ly = pickle.dumps(self.layer_stack)
        with open(file, 'wb') as f:
            f.write(ly)

    def load(self, file):
        """
        读取模型
        :param file:
        :return:
        """
        with open(file, 'rb') as f:
            ly_stack = pickle.load(f)

        network = model()
        network.layer_stack = ly_stack
        return network


if __name__ == '__main__':
    data = datasets.load_iris().data
    label = datasets.load_iris().target
    label = one_hot(label, 3)
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.6, random_state=1)

    def iris_data_loader(x, y, bs, e):
        np.random.seed(cft.seed)
        np.random.shuffle(x)
        np.random.seed(cft.seed)
        np.random.shuffle(y)

        length = len(x)

        for batch in tqdm(range(0, length, bs), desc=f"{e}"):
            batch_x = x[batch:batch+bs]
            batch_y = y[batch:batch+bs]
            yield batch_x, batch_y

    model_p = model()

    l1 = linear(4, 8)
    # l2 = linear(8, 6)
    l3 = linear(8, 3)

    model_p.add(l1, activation_f="relu")
    # model_p.add(l2, activation_f="sigmoid")
    model_p.add(l3, activation_f="softmax")
    model_p.compile(optimizer="SGD", init_parameters="normal", loss="cross_entropy", rand_seed=cft.seed)
    model_p.summary()
    for epoch in range(epochs):
        train_loader = iris_data_loader(x_train, y_train, batch_size, epoch)

        start_time = time.time()        # 训练开始的时间
        loss, acc = model_p.fit(train_loader)
        end_time = time.time()          # 训练结束的时间

        acc_all.append(acc)
        loss_all.append(loss)
        print(f"train_loss={loss}, train_acc={acc}, train_time={end_time - start_time}")
    pre_test = model_p.predict_classes(x_test)
    acc_test = accuracy_score(pre_test, y_test)
    print("--"*50)
    print(f"test accuracy = {acc_test}")
    print("--"*50)
    model_plot(loss_all, acc_all, epochs)
    model_p.save("iris_net")
