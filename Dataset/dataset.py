# -*- coding: utf-8 -*-
# @Time    : 2021/5/13
# @Author  : Ke
# 参考 https://blog.csdn.net/panrenlong/article/details/81736754 中读取数据集的操作，并在其基础上对数据做预处理
import numpy as np
import struct
import matplotlib.pyplot as plt
from cnn.Utils.utils import one_hot
from PIL import Image, ImageOps

__all__ = ["load_train_images", "load_train_labels", "load_test_images", "load_test_labels", "img_reshape"]


# 训练集文件
train_images_idx3_ubyte_file = r'../Dataset/mnist_dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = r'../Dataset/mnist_dataset/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = r'../Dataset/mnist_dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = r'../Dataset/mnist_dataset/t10k-labels.idx1-ubyte'
# 随机种子
seed = 123
# mnist数据集标准化参数，官方给的
normalize_mnist = {"mean": 0.1307, "std": 0.3081}
# 是否对数据集标准化，归一化将数据值调整到[0, 1]，标准化将归一化的数据调整到[-1, 1]
normalize = False


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    # print(offset)
    fmt_image = '>' + str(image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    # print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
            # print(offset)
            pass
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    images.reshape(num_images, 1, num_rows, num_cols)

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            # print ('已解析 %d' % (i + 1) + '张')
            pass
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    train_x = decode_idx3_ubyte(idx_ubyte_file)
    train_x /= 255.
    if normalize:
        train_x = (train_x - normalize_mnist["mean"]) / normalize_mnist["std"]
    np.random.seed(seed)
    np.random.shuffle(train_x)
    return train_x


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    train_y = decode_idx1_ubyte(idx_ubyte_file)
    np.random.seed(seed)
    np.random.shuffle(train_y)
    return train_y


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    test_x = decode_idx3_ubyte(idx_ubyte_file)
    test_x /= 255.
    if normalize:
        test_x = (test_x - normalize_mnist["mean"]) / normalize_mnist["std"]
    np.random.seed(seed)
    np.random.shuffle(test_x)
    return test_x


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    test_y = decode_idx1_ubyte(idx_ubyte_file)
    np.random.seed(seed)
    np.random.shuffle(test_y)
    return test_y


def img_reshape(img):
    bs, h, w = img.shape
    imre = np.zeros((bs, 1, h, w))
    for i in range(bs):
        imtemp = Image.fromarray(np.reshape(img[i], (28, 28)))
        # imtemp.rotate((random.random() - 0.5) * 30)
        imre[i] = np.array(imtemp)
    return imre


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    print(train_labels)
    print(img_reshape(train_images).shape)
    img = train_images[0]
    # print(img)
    plt.imshow(img, cmap='gray')
    plt.show()
