import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils

class LystoDataset(Dataset):

    def __init__(self, filepath=None,
                 transform=None,
                 train=True,
                 kfold=10,
                 interval=20,
                 size=32,
                 num_of_imgs=0):
        """
        :param filepath:    hdf5数据文件路径
        :param transform:   数据预处理方式
        :param train:       训练集 / 验证集，默认为训练集
        :param kfold:       k 折交叉验证的参数，数据集每隔 k 份抽取 1 份作为验证集，默认值为 10
        :param interval:    在切片上选取 patch 的间隔，默认值为 20px
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
        self.visualize = False
        self.organs = []            # 全切片来源，array ( 20000 )
        self.images = []            # array ( 20000 * 299 * 299 * 3 )
        self.labels = []            # 图像中的阳性细胞数目，array ( 20000 )
        self.patchIDX = []          # 每个patch对应的图像编号，array ( 20000 * n )
        self.patches_grid = []           # 每张图像中选取的像素 patch 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = interval
        self.size = size

        patchIDX = -1
        for i, (organ, img, label) in enumerate(zip(f['organ'], f['x'], f['y'])):

            # 调试用代码
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            if (self.train and (i + 1) % self.kfold == 0) or (not self.train and (i + 1) % self.kfold != 0):
                continue

            patchIDX += 1
            self.organs.append(organ)
            self.images.append(img)
            self.labels.append(label)
            p = get_patches(img, self.interval, self.size)
            self.patches_grid.extend(p) # 获取 patch
            self.patchIDX.extend([patchIDX] * len(p)) # 每个 patch 对应的 slide 标签

        assert len(self.labels) == len(self.images)

        self.mode = None
        self.transform = transform

    def setmode(self, mode):
        self.mode = mode

    # def visualize_bboxes(self):
    #     self.visualize = True

    def make_train_data(self, idxs):
        # 用于 mode 2，制作训练用数据集
        # 当 patch 对应的切片的 label 为 n 时标签为 1 ，否则为 0
        self.train_data = [(self.patchIDX[i], self.patches_grid[i],
                           0 if self.labels[self.patchIDX[i]] == 0 else 1) for i in idxs]
        # if shuffle:
        #     self.train_data = random.sample(self.train_data, len(self.train_data))

        pos = 0
        for _, _, label in self.train_data:
            pos += label

        # 返回正负样本数目
        return pos, len(self.train_data) - pos

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        # top-k 选取模式 (patch mode)
        if self.mode == 1:
            (x, y) = self.patches_grid[idx]
            patch = self.images[self.patchIDX[idx]][x:x + self.size - 1, y:y + self.size - 1]
            if self.transform:
                patch = self.transform(patch)

            label = self.labels[self.patchIDX[idx]]
            return patch, label

        # alternative training mode
        elif self.mode == 2:
            # Get slides
            slide = self.images[idx]
            label_cls = 0 if self.labels[idx] == 0 else 1
            label_reg = self.labels[idx]

            if self.visualize:
                plt.imshow(slide)

            # Get patches
            patches = []
            patch_grids = []
            patch_labels = []
            for patchIDX, (x, y), label in self.train_data:
                if patchIDX == idx:
                    patch = self.images[patchIDX][x:x + self.size - 1, y:y + self.size - 1]
                    patches.append(patch)
                    patch_grids.append((x, x + self.size - 1, y, y + self.size - 1))
                    patch_labels.append(label)

                    if self.visualize:
                        plt.gca().add_patch(
                            plt.Rectangle((x, y), x + self.size - 1, y + self.size - 1,
                                          fill=False, edgecolor='red' if label == 0 else 'deepskyblue', linewidth=1)
                        )

            # patch visualize testing
            if self.visualize:
                plt.savefig('test/img{}.png'.format(idx))
                plt.close()

            #     slide_tensor = torch.from_numpy(slide.transpose((2, 0, 1))).contiguous()
            #     utils.draw_bounding_boxes(slide_tensor, torch.tensor(patch_grids),
            #                               labels=['neg' if lbl == 0 else 'pos' for lbl in patch_labels],
            #                               colors=list(cycle('red')))
            #     utils.save_image(slide, "test/img{}.png".format(idx))
            #     print("Image is saved.")

            if self.transform:
                patches = [self.transform(patch) for patch in patches]
                slide = self.transform(slide)

            return (slide.unsqueeze(0), torch.stack(patches, dim=0)), \
                   (label_cls, label_reg, [], torch.tensor(patch_labels))

        # patch-only training mode
        elif self.mode == 3:
            patchIDX, (x, y), label = self.train_data[idx]
            patch = self.images[patchIDX][x:x + self.size - 1, y:y + self.size - 1]
            if self.transform:
                patch = self.transform(patch)
            return patch, label

        # slide-only training & validating mode
        elif self.mode == 4:
            slide = self.images[idx]
            label_cls = 0 if self.labels[idx] == 0 else 1
            label_reg = self.labels[idx]

            if self.transform:
                slide = self.transform(slide)
            return slide, label_cls, label_reg, []

        else:
            raise Exception("Something wrong in setmode.")


    def __len__(self):

        if self.mode == 1:
            return len(self.patchIDX)
        elif self.mode == 2:
            return len(self.images)
        elif self.mode == 3:
            return len(self.train_data)
        elif self.mode == 4:
            return len(self.labels)
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
        self.patchIDX = []          # 每个patch对应的图像编号，array ( 20000 * n )
        self.patches_grid = []      # 每张图像中选取的像素 patch 的左上角坐标点，array ( 20000 * n * 2 )
        self.interval = interval
        self.size = size

        patchIDX = -1
        for i, (organ, img) in enumerate(zip(f['organ'], f['x'])):

            # TODO: 调试用代码，实际代码不包含 num_of_imgs 参数及以下两行
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            patchIDX += 1
            self.organs.append(organ)
            self.images.append(img)
            p = get_patches(img, self.interval, self.size)
            self.patches_grid.extend(p) # 获取 patch
            self.patchIDX.extend([patchIDX] * len(p))

        self.transform = transform

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        (x, y) = self.patches_grid[idx]
        patch = self.images[self.patchIDX[idx]][x:x + self.size - 1, y:y + self.size - 1]
        if self.transform is not None:
            patch = self.transform(patch)

        return patch

    def __len__(self):
        return len(self.patchIDX)


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
    imageSet = LystoDataset(filepath="data/training.h5", interval=150, size=32, num_of_imgs=51)
    imageSet_val = LystoDataset(filepath="data/training.h5", interval=150, size=32, num_of_imgs=51, train=False)
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
    print("Grids of patches: {}".format(imageSet.patches_grid[0]))

    print("The first validation image: ")
    plt.imshow(imageSet_val.images[0])
    plt.show()
    print("Slide: {0}\nLabel: {1}".format(imageSet_val.organs[0], imageSet_val.labels[0]))
    print("Grids of patches: {}".format(imageSet_val.patches_grid[0]))
