"""
图像分类数据集：
    - Imagenet, 下载链接 http://www.image-net.org/
    - CIFAR（包括CIFAR-10和CIFAR-100）, 下载链接 http://www.cs.toronto.edu/~kriz/cifar.html
    - Flowers-17, 下载链接 https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
    - Animals, 下载链接 https://www.kaggle.com/alessiocorrado99/animals10
    - Stanford Cars, 下载链接 http://ai.stanford.edu/~jkrause/cars/car_dataset.html
    - Facial Expression Recognition Challenge, 下载链接 https://www.kaggle.com/debanga/facial-expression-recognition-challenge
    - Indoor, 下载链接 https://interiornet.org/
    ......
"""

__all__ = ["mnist", "fashion_mnist"]

# mnist数据集
mnist = {
    "train_images_path": r'../Dataset/mnist_dataset/train-images.idx3-ubyte',
    "train_labels_path": r'../Dataset/mnist_dataset/train-labels.idx1-ubyte',
    "test_images_path": r'../Dataset/mnist_dataset/t10k-images.idx3-ubyte',
    "test_labels_path": r'../Dataset/mnist_dataset/t10k-labels.idx1-ubyte'
}

# fashion mnist数据集
fashion_mnist = {
    "train_images_path": r'../Dataset/fashion_mnist/train-images-idx3-ubyte',
    "train_labels_path": r'../Dataset/fashion_mnist/train-labels-idx1-ubyte',
    "test_images_path": r'../Dataset/fashion_mnist/t10k-images-idx3-ubyte',
    "test_labels_path": r'../Dataset/fashion_mnist/t10k-labels-idx1-ubyte'
}