import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


class LystoDataset(Dataset):

    def __init__(self, filepath=None,
                 transform=None,
                 train=True):

        if filepath:
            f = h5py.File(filepath, 'r')
        else:
            raise Exception("Invalid data file.")

        self.organs = []        # 全切片来源，array (20000)
        self.images = []        # array ( 20000 * 299 * 299 * 3 )
        self.labels = []        # 图像中的阳性细胞数目，array ( 20000 )
        self.imageIDX = []      # 每个patch对应的图像号，array ( 20000 * n )
        self.patches = []       # 每张图像中选取的像素 patch 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = 3       # 取实例的像素间隔
        self.size = 32          # 一个实例的大小

        for i, (organ, img, label) in enumerate(zip(f['organ'], f['x'], f['y'])):
            self.organs.append(organ)
            self.images.append(img)
            # self.labels.append(label)
            self.labels.append(1 if label != 0 else 0) # TODO: 暂时把标签当作非计数式标签处理
            p = get_patches(img, self.interval, self.size)
            self.patches.append(p) # 获取 32 * 32 的实例
            self.imageIDX.extend([i] * len(p))

        self.mode = None
        self.transform = transform

    def setmode(self, mode):
        self.mode = mode

    def make_train_data(self, idxs, shuffle=True): # 用于 mode 2，制作训练用数据集
        self.train_data = [(self.imageIDX[i], self.patches[i // len(self.patches[0])][i % len(self.patches[0])],
                            self.labels[self.imageIDX[i]]) for i in idxs]
        if shuffle:
            self.train_data = random.sample(self.train_data, len(self.train_data))

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        if self.mode == 1: # top-k 选取模式
            (x, y) = self.patches[idx // len(self.patches[0])][idx % len(self.patches[0])]
            patch = self.images[self.imageIDX[idx]][x:x + self.size - 1, y:y + self.size - 1]
            if self.transform is not None:
                patch = self.transform(patch)

            label = self.labels[self.imageIDX[idx]]

        elif self.mode == 2: # 训练数据模式
            imageIDX, patch_grid, label = self.train_data[idx]
            (x, y) = patch_grid
            patch = self.images[imageIDX][x:x + self.size - 1, y:y + self.size - 1]
            if self.transform is not None:
                patch = self.transform(patch)

        else:
            raise Exception("Something wrong in setmode.")

        return patch, label

    def __len__(self):
        if self.mode == 1:
            return len(self.imageIDX)
        elif self.mode == 2:
            return len(self.train_data)
        else:
            raise Exception("Something wrong in setmode.")


def get_patches(image, interval=3, size=32):
    """
    在每张图片上生成小patch实例。
    :param image: 输入图片矩阵，299 x 299 x 3
    :param interval: 取patch坐标点的间隔
    :param size: 单个patch的大小
    """

    patches = []
    for x in np.arange(0, image.shape[0] - size + 1, interval):
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            patches.append((x, y))  # n x 2

    # for i, (x, y) in enumerate(patches):
    #     patches[i] = image[x:x+size-1, y:y+size-1]  # n x 32 x 32 x 3

    return patches


if __name__ == '__main__':

    imageSet = LystoDataset(filepath="D:/LYSTO/training.h5")
    loader = DataLoader(imageSet, batch_size=1, shuffle=False)

    imageSet.setmode(1)
    for idx, data in enumerate(loader):
        print('Dry Run : [{}/{}]\r'.format(idx + 1, len(loader.dataset)))
    print("Length of dataset: {}".format(len(loader.dataset)))

    # 查看第一张图片
    print("The first image: ")
    plt.imshow(imageSet.images[0])
    print("Slide: {0}\nLabel: {1}".format(imageSet.organs[0], imageSet.labels[0]))
    print("Grids of patches: {}".format(imageSet.patches[0]))
    print("Length of dataset: {}".format(len(loader.dataset)))

