# -*- coding: utf-8 -*-
# @Time    : 2021/5/14
# @Author  : Ke
import numpy as np
import struct
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cnn.Configurations.configurations as cft
from cnn.Utils.utils import *

__all__ = ["data_augmentation", "data_loader"]


# 训练集文件
train_images_path = r'../Dataset/mnist_dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_path = r'../Dataset/mnist_dataset/train-labels.idx1-ubyte'

# 测试集文件
test_images_path = r'../Dataset/mnist_dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_path = r'../Dataset/mnist_dataset/t10k-labels.idx1-ubyte'


def data_augmentation(batch_image):
    """
    数据增强
    :param batch_image: 一个batch的图像
    :return: 增强后的数据，数据的维度也变为(bs, channel, h, w)
    """
    bs, _, _ = batch_image.shape
    imre = np.zeros((bs, 1, 28, 28))
    idx = np.arange
    for i in range(bs):
        imtemp = Image.fromarray(np.reshape(batch_image[i], (28, 28)))
        # imtemp.rotate((random.random() - 0.5) * 30)
        imre[i] = np.array(imtemp)

    return imre


def data_loader(image, label, bs, is_shuffle=True):
    with open(label, 'rb') as lb:
        # file header [0,9]
        m, n = struct.unpack('>II', lb.read(8))
        labels = np.fromfile(lb, dtype=np.uint8)

    with open(image, 'rb') as im:
        m, n, r, c = struct.unpack('>IIII', im.read(16))
        images = np.fromfile(im, dtype=np.uint8).reshape(len(labels), 28, 28)

    # length = len(labels)
    labels = one_hot(labels, 10)

    while True:
        # 打乱数据集
        if is_shuffle:
            np.random.seed(cft.seed)
            np.random.shuffle(images)
            np.random.seed(cft.seed)
            np.random.shuffle(labels)

        for batch_idx in range(0, n, bs):
            # print(batch_idx)
            batch_label = labels[batch_idx:batch_idx + bs]
            batch_image = images[batch_idx:batch_idx + bs].astype('float32')
            # normalize
            batch_image = data_augmentation(batch_image / 255.0)

            yield batch_image, batch_label


if __name__ == '__main__':
    batch_size = 128
    _iter = data_loader(train_images_path, train_labels_path, batch_size)
    train_img, train_labels = _iter.__next__()
    print(train_img.shape)
    p_img = train_img[0]
    print(train_labels[0])
    plt.imshow(p_img[0], cmap='gray')
    plt.show()
