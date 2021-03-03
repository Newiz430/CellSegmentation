import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader


class LystoDataset(Dataset):

    def __init__(self, filepath=None,
                 transform=None,
                 train=True,
                 kfold=10,
                 interval=10,
                 size=32,
                 num_of_imgs=0):
        """
        :param filepath:    hdf5数据文件路径
        :param transform:   数据预处理方式
        :param train:       训练集 / 验证集，默认为训练集
        :param kfold:       k 折交叉验证的参数，数据集每隔 k 份抽取 1 份作为验证集，默认值为 10
        :param interval:    在切片上选取 patch 的间隔，默认值为 10px
        :param size:        一个 patch 的边长，默认值为 32px
        :param num_of_imgs: 调试程序用参数，表示用数据集的前 n 张图片构造数据集，设为 0 使其不起作用
        """

        if filepath:
            f = h5py.File(filepath, 'r')
        else:
            raise Exception("Invalid data file.")

        if kfold <= 0:
            raise Exception("Invalid k-fold cross-validation argument.")

        self.train = train
        self.kfold = kfold
        self.organs = []            # 全切片来源，array ( 20000 )
        self.images = []            # array ( 20000 * 299 * 299 * 3 )
        self.labels = []            # 图像中的阳性细胞数目，array ( 20000 )
        self.imageIDX = []          # 每个patch对应的图像编号，array ( 20000 * n )
        self.patches = []           # 每张图像中选取的像素 patch 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = interval
        self.size = size

        imageIDX = -1
        for i, (organ, img, label) in enumerate(zip(f['organ'], f['x'], f['y'])):

            # TODO: 调试用代码，实际代码不包含 num_of_imgs 参数及以下两行
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            if (self.train and (i + 1) % self.kfold == 0) or (not self.train and (i + 1) % self.kfold != 0):
                continue

            imageIDX += 1
            self.organs.append(organ)
            self.images.append(img)
            # self.labels.append(label)
            self.labels.append(1 if label != 0 else 0) # TODO: 暂时把标签当作非计数式标签处理
            p = get_patches(img, self.interval, self.size)
            self.patches.extend(p) # 获取 32 * 32 的实例
            self.imageIDX.extend([imageIDX] * len(p))

        self.mode = None
        self.transform = transform

    def setmode(self, mode):
        self.mode = mode

    def make_train_data(self, idxs, shuffle=True): # 用于 mode 2，制作训练用数据集
        self.train_data = [(self.imageIDX[i], self.patches[i],
                            self.labels[self.imageIDX[i]]) for i in idxs]
        if shuffle:
            self.train_data = random.sample(self.train_data, len(self.train_data))

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        if self.mode == 1: # top-k 选取模式
            (x, y) = self.patches[idx]
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


class LystoTestset(Dataset):

    def __init__(self, filepath=None,
                 transform=None,
                 interval=10,
                 size=32,
                 num_of_imgs=0):
        """
        :param filepath:    hdf5数据文件路径
        :param transform:   数据预处理方式
        :param interval:    在切片上选取 patch 的间隔，默认值为 10px
        :param size:        一个 patch 的边长，默认值为 32px
        :param num_of_imgs: 调试程序用参数，表示用数据集的前 n 张图片构造数据集，设为 0 使其不起作用
        """

        if filepath:
            f = h5py.File(filepath, 'r')
        else:
            raise Exception("Invalid data file.")

        self.organs = []            # 全切片来源，array ( 20000 )
        self.images = []            # array ( 20000 * 299 * 299 * 3 )
        self.imageIDX = []          # 每个patch对应的图像编号，array ( 20000 * n )
        self.patches = []           # 每张图像中选取的像素 patch 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = interval
        self.size = size

        imageIDX = -1
        for i, (organ, img) in enumerate(zip(f['organ'], f['x'])):

            # TODO: 调试用代码，实际代码不包含 num_of_imgs 参数及以下两行
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            imageIDX += 1
            self.organs.append(organ)
            self.images.append(img)
            p = get_patches(img, self.interval, self.size)
            self.patches.extend(p) # 获取 32 * 32 的实例
            self.imageIDX.extend([imageIDX] * len(p))

        self.transform = transform

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        (x, y) = self.patches[idx]
        patch = self.images[self.imageIDX[idx]][x:x + self.size - 1, y:y + self.size - 1]
        if self.transform is not None:
            patch = self.transform(patch)

        return patch

    def __len__(self):
        return len(self.imageIDX)


def get_patches(image, interval=10, size=32):
    """
    在每张图片上生成小 patch 实例。
    :param image: 输入图片矩阵，299 x 299 x 3
    :param interval: 取patch坐标点的间隔
    :param size: 单个patch的大小
    """

    patches = []
    for x in np.arange(0, image.shape[0] - size + 1, interval):
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            patches.append((x, y))  # n x 2

    return patches


if __name__ == '__main__':

    batch_size = 2
    imageSet = LystoDataset(filepath="LYSTO/training.h5", interval=150, size=32, num_of_imgs=51)
    imageSet_val = LystoDataset(filepath="LYSTO/training.h5", interval=150, size=32, num_of_imgs=51, train=False)
    train_loader = DataLoader(imageSet, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(imageSet_val, batch_size=batch_size, shuffle=False)

    imageSet.setmode(1)
    imageSet_val.setmode(1)
    for idx, data in enumerate(train_loader):
        print('Dry Run : [{}/{}]\r'.format(idx + 1, len(train_loader.dataset) // batch_size))
    print("Length of dataset: {}".format(len(train_loader.dataset)))
    for idx, data in enumerate(val_loader):
        print('Dry Run : [{}/{}]\r'.format(idx + 1, len(val_loader.dataset) // batch_size))
    print("Length of dataset: {}".format(len(val_loader.dataset)))

    # 查看第一张图片
    print("The first training image: ")
    plt.imshow(imageSet.images[0])
    plt.show()
    print("Slide: {0}\nLabel: {1}".format(imageSet.organs[0], imageSet.labels[0]))
    print("Grids of patches: {}".format(imageSet.patches[0]))

    print("The first validation image: ")
    plt.imshow(imageSet_val.images[0])
    plt.show()
    print("Slide: {0}\nLabel: {1}".format(imageSet_val.organs[0], imageSet_val.labels[0]))
    print("Grids of patches: {}".format(imageSet_val.patches[0]))
