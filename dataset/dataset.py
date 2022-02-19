import collections
import os
import sys
from tqdm import tqdm

import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
from skimage import io
from openslide import OpenSlide

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils

Image.MAX_IMAGE_PIXELS = None
patch_size = np.array([299, 299])

class LystoDataset(Dataset):

    def __init__(self, filepath=None, tile_size=None, interval=None, train=True, augment=False, kfold=10, num_of_imgs=0,
                 _stacking_init=False):
        """
        :param filepath:    hdf5数据文件路径
        :param train:       训练集 / 验证集，默认为训练集
        :param kfold:       k 折交叉验证的参数，数据集每隔 k 份抽取 1 份作为验证集，默认值为 10
        :param interval:    选取 tile 的间隔步长
        :param tile_size:   一个 tile 的边长
        :param num_of_imgs: 调试程序用参数，表示用数据集的前 n 张图片构造数据集，设为 0 使其不起作用
        """

        super(LystoDataset, self).__init__()

        if not _stacking_init:
            if os.path.exists(filepath):
                f = h5py.File(filepath, 'r')
            else:
                raise FileNotFoundError("Invalid data directory.")

            if kfold is not None and kfold <= 0:
                raise Exception("Invalid k-fold cross-validation argument.")
            else:
                self.kfold = kfold

        self.train = train
        # self.visualize = False
        self.organs = []  # 全切片来源，list ( 20000 )
        self.images = []  # list ( 20000 * 299 * 299 * 3 )
        self.labels = []  # 图像中的阳性细胞数目，list ( 20000 )
        self.cls_labels = []  # 按数目把图像分为 7 类，存为类别标签
        self.transformIDX = []  # 数据增强的类别，list ( 20000 )
        self.tileIDX = []  # 每个 tile 对应的图像编号，list ( 20000 * n )
        self.tiles_grid = []  # 每张图像中选取的像素 tile 的左上角坐标点，list ( 20000 * n * 2 )
        self.interval = interval
        self.tile_size = tile_size

        self.augment = augment
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1)
            ])
        ]
        self.transform = [transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])] + [transforms.Compose([
            transforms.ToTensor(),
            aug,
            # transforms.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.4,
            #     hue=0.05,
            # ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) for aug in augment_transforms]

        self.image_size = patch_size
        self.mode = None

        if not _stacking_init:
            def store_data(transidx=0):

                nonlocal organ, img, label, tileIDX, augment_transforms
                assert transidx <= len(augment_transforms), "Not enough transformations for image augmentation. "

                self.organs.append(organ)
                self.images.append(img)
                self.labels.append(label)
                cls_label = categorize(label)
                self.cls_labels.append(cls_label)
                self.transformIDX.append(transidx)

                if self.interval is not None and self.tile_size is not None:
                    t = get_tiles(img, self.interval, self.tile_size)
                    self.tiles_grid.extend(t)  # 获取 tiles
                    self.tileIDX.extend([tileIDX] * len(t))  # 每个 tile 对应的 image 标签

                return cls_label

            tileIDX = -1
            for i, (organ, img, label) in enumerate(tqdm(zip(f['organ'], f['x'], f['y']), desc="loading images")):

                # 调试用代码
                if num_of_imgs != 0 and i == num_of_imgs:
                    break

                if self.kfold is not None:
                    if (self.train and (i + 1) % self.kfold == 0) or (not self.train and (i + 1) % self.kfold != 0):
                        continue

                tileIDX += 1

                cls_label = store_data()
                if self.train and self.augment and cls_label >= 3:
                    for i in range(1, 4):
                        store_data(i)

            assert len(self.labels) == len(self.images), "Mismatched number of labels and images."

    def setmode(self, mode):
        """
        mode 1: instance inference mode, for top-k sampling -> tile (sampled from images), label
        mode 2: alternative training mode (alternate tile training + image training per batch iteration)
                -> (image, tiles sampled from which), (class, number, binary tile labels)
        mode 3: tile-only training mode -> tile (from top-k training data), label
        mode 4: image validating mode -> 3d image, class, number
        mode 5: image-only training mode -> 4d image, class, number
        """
        self.mode = mode

    # def visualize_bboxes(self):
    #     self.visualize = True

    def make_train_data(self, idxs):
        # 制作 tile mode 训练用数据集，当 tile 对应的图像的 label 为 n 时标签为 1 ，否则为 0
        self.train_data = [(self.tileIDX[i], self.tiles_grid[i],
                            0 if self.labels[self.tileIDX[i]] == 0 else 1) for i in idxs]
        # if shuffle:
        #     self.train_data = random.sample(self.train_data, len(self.train_data))

        pos = 0
        for _, _, label in self.train_data:
            pos += label

        # 返回正负样本数目
        return pos, len(self.train_data) - pos

    def __getitem__(self, idx):

        # organ = self.organs[idx]

        # top-k tile sampling mode
        if self.mode == 1:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile inference. "

            (x, y) = self.tiles_grid[idx]
            tile = self.images[self.tileIDX[idx]][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform[self.transformIDX[self.tileIDX[idx]]](tile)

            label = self.labels[self.tileIDX[idx]]
            return tile, label

        # alternative training mode
        elif self.mode == 2:
            assert len(self.tiles_grid) > 0, \
                "Dataset tile size and interval have to be settled for alternative training. "

            # Get images
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            # if self.visualize:
            #     plt.imshow(image)

            # Get tiles
            tiles = []
            tile_grids = []
            tile_labels = []
            for tileIDX, (x, y), label in self.train_data:
                if tileIDX == idx:
                    tile = self.images[tileIDX][x:x + self.tile_size, y:y + self.tile_size]
                    tiles.append(tile)
                    tile_grids.append((x, x + self.tile_size, y, y + self.tile_size))
                    tile_labels.append(label)

                    # if self.visualize:
                    #     plt.gca().add_patch(
                    #         plt.Rectangle((x, y), x + self.size, y + self.size,
                    #                       fill=False, edgecolor='red' if label == 0 else 'deepskyblue', linewidth=1)
                    #     )

            # # tile visualize testing
            # if self.visualize:
            #     plt.savefig('test/img{}.png'.format(idx))
            #     plt.close()

            # # 画边界框（有问题）
            #     image_tensor = torch.from_numpy(tile.transpose((2, 0, 1))).contiguous()
            #     utils.draw_bounding_boxes(image_tensor, torch.tensor(tile_grids),
            #                               labels=['neg' if lbl == 0 else 'pos' for lbl in tile_labels],
            #                               colors=list(cycle('red')))
            #     utils.save_image(image, "test/img{}.png".format(idx))
            #     print("Image is saved.")

            tiles = [self.transform[self.transformIDX[self.tileIDX[idx]]](tile) for tile in tiles]
            image = self.transform[self.transformIDX[idx]](image)

            return (image.unsqueeze(0), torch.stack(tiles, dim=0)), \
                   (label_cls, label_reg, torch.tensor(tile_labels))

        # tile-only training mode
        elif self.mode == 3:
            assert len(
                self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile-mode training. "

            tileIDX, (x, y), label = self.train_data[idx]
            tile = self.images[tileIDX][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform[self.transformIDX[self.tileIDX[idx]]](tile)
            return tile, label

        # image validating mode
        elif self.mode == 4:
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            image = self.transform[self.transformIDX[idx]](image)
            return image, label_cls, label_reg

        # image-only training mode
        elif self.mode == 5:
            image = self.images[idx]
            # label_cls = 0 if self.labels[idx] == 0 else 1
            label_cls = self.cls_labels[idx]
            label_reg = self.labels[idx]

            image = self.transform[self.transformIDX[idx]](image)
            # for image-only training, images need to be unsqueezed
            return image.unsqueeze(0), label_cls, label_reg

        else:
            raise Exception("Something wrong in setmode.")

    def __len__(self):

        assert self.mode is not None, "Something wrong in setmode."

        if self.mode == 1:
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "
            return len(self.tileIDX)
        elif self.mode == 2:
            return len(self.images)
        elif self.mode == 3:
            return len(self.train_data)
        else:
            return len(self.labels)


class EnsembleSet:

    def __init__(self, filepath=None, augment=False, k: int = 10):

        data = LystoDataset(filepath, kfold=None, augment=augment)
        self.k = k
        images = np.array_split(np.asarray(data.images), self.k)
        labels = np.array_split(np.asarray(data.labels), self.k)
        organs = np.array_split(np.asarray(data.organs), self.k)
        cls_labels = np.array_split(np.asarray(data.cls_labels), self.k)
        transformations = np.array_split(np.asarray(data.transformIDX), self.k)

        self.training_sets = [LystoDataset(kfold=None, _stacking_init=True)] * self.k
        self.validating_sets = [LystoDataset(kfold=None, _stacking_init=True)] * self.k

        for i in range(self.k):
            for j in range(self.k):

                if j == 0:
                    self.training_sets[i].images = list(np.concatenate(images[1:]))
                    self.training_sets[i].labels = list(np.concatenate(labels[1:]))
                    self.training_sets[i].organs = list(np.concatenate(organs[1:]))
                    self.training_sets[i].cls_labels = list(np.concatenate(cls_labels[1:]))
                    self.training_sets[i].transformIDX = list(np.concatenate(transformations[1:]))
                elif j == self.k - 1:
                    self.training_sets[i].images = list(np.concatenate(images[:-1]))
                    self.training_sets[i].labels = list(np.concatenate(labels[:-1]))
                    self.training_sets[i].organs = list(np.concatenate(organs[:-1]))
                    self.training_sets[i].cls_labels = list(np.concatenate(cls_labels[:-1]))
                    self.training_sets[i].transformIDX = list(np.concatenate(transformations[:-1]))
                else:
                    self.training_sets[i].images = list(np.concatenate(images[:j])) + \
                                                   list(np.concatenate(images[j + 1:]))
                    self.training_sets[i].labels = list(np.concatenate(labels[:j])) + \
                                                   list(np.concatenate(labels[j + 1:]))
                    self.training_sets[i].organs = list(np.concatenate(organs[:j])) + \
                                                   list(np.concatenate(organs[j + 1:]))
                    self.training_sets[i].cls_labels = list(np.concatenate(cls_labels[:j])) + \
                                                       list(np.concatenate(cls_labels[j + 1:]))
                    self.training_sets[i].transformIDX = list(np.concatenate(transformations[:j])) + \
                                                         list(np.concatenate(transformations[j + 1:]))
                self.validating_sets[i].images = images[j]
                self.validating_sets[i].labels = labels[j]
                self.validating_sets[i].organs = organs[j]
                self.validating_sets[i].cls_labels = cls_labels[j]
                self.validating_sets[i].transformIDX = transformations[j]

    def setmode(self, train, mode):
        if train:
            for i in range(self.k):
                self.training_sets[i].setmode(mode)
        else:
            for i in range(self.k):
                self.validating_sets[i].setmode(mode)

    def get_loader(self, train, idx, **kwargs):
        if train:
            return DataLoader(self.training_sets[idx], shuffle=True, pin_memory=True, **kwargs)
        else:
            return DataLoader(self.validating_sets[idx], shuffle=False, pin_memory=True, **kwargs)


class LystoTestset(Dataset):

    def __init__(self, filepath, tile_size=None, interval=None, num_of_imgs=0):
        """
        :param filepath:    数据文件路径 (hdf5)
        :param interval:    选取 tile 的间隔步长
        :param tile_size:   一个 tile 的边长
        :param num_of_imgs: 调试程序用参数，表示用数据集的前 n 张图片构造数据集，设为 0 使其不起作用
        """

        super(LystoTestset, self).__init__()

        if os.path.exists(filepath):
            f = h5py.File(filepath, 'r')
        else:
            raise FileNotFoundError("Invalid data directory.")

        self.organs = []  # 全切片来源，list ( 20000 )
        self.images = []  # list ( 20000 * 299 * 299 * 3 )
        self.tileIDX = []  # 每个 tile 对应的图像编号，list ( 20000 * n )
        self.tiles_grid = []  # 每张图像中选取的像素 tile 的左上角坐标点，list ( 20000 * n * 2 )
        self.interval = interval
        self.tile_size = tile_size

        tileIDX = -1
        for i, (organ, img) in enumerate(tqdm(zip(f['organ'], f['x']), desc="loading images")):

            # TODO: 调试用代码，实际代码不包含 num_of_imgs 参数及以下两行
            if num_of_imgs != 0 and i == num_of_imgs:
                break

            tileIDX += 1
            self.organs.append(organ)
            self.images.append(img)

            if self.interval is not None and self.tile_size is not None:
                t = get_tiles(img, self.interval, self.tile_size)
                self.tiles_grid.extend(t)
                self.tileIDX.extend([tileIDX] * len(t))

        self.image_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.mode = None

    def setmode(self, mode):
        """
        mode "tile":  instance mode, used in pseudo-mask heatmap generation -> tile (sampled from images)
        mode "image": image assessment mode, used in cell counting -> image
        """
        self.mode = mode

    def __getitem__(self, idx):
        # test_tile
        if self.mode == "tile":
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "

            # organ = self.organs[idx]
            (x, y) = self.tiles_grid[idx]
            tile = self.images[self.tileIDX[idx]][x:x + self.tile_size, y:y + self.tile_size]
            tile = self.transform(tile)

            return tile

        # test_count
        elif self.mode == "image":
            image = self.images[idx]
            image = self.transform(image)

            return image

        else:
            raise Exception("Something wrong in setmode.")

    def __len__(self):
        if self.mode == "tile":
            assert len(self.tiles_grid) > 0, "Dataset tile size and interval have to be settled for tile mode. "
            return len(self.tileIDX)
        elif self.mode == "image":
            return len(self.images)
        else:
            raise Exception("Something wrong in setmode.")


class Maskset(Dataset):

    def __init__(self, filepath, mask_data, num_of_imgs=0):

        super(Maskset, self).__init__()
        assert type(mask_data) in [np.ndarray, str], "Invalid data type. "

        if filepath:
            f = h5py.File(filepath, 'r')
        else:
            raise FileNotFoundError("Invalid data file.")

        self.organs = []
        self.images = []
        self.masks = []
        self.labels = []

        for i, (organ, img, label) in enumerate(tqdm(zip(f['organ'], f['x'], f['y']), desc="loading images")):
            if num_of_imgs != 0 and i == num_of_imgs:
                break
            self.organs.append(organ)
            self.images.append(img)
            self.labels.append(label)

        if isinstance(mask_data, str):

            for file in sorted(os.listdir(os.path.join(mask_data, 'mask'))):
                if num_of_imgs != 0 and len(self.masks) == len(self.images):
                    break
                img = io.imread(os.path.join(mask_data, 'mask', file))
                self.masks.append(img)

        else:
            self.masks = [torch.from_numpy(np.uint8(md)) for md in mask_data]
            if num_of_imgs != 0:
                self.masks = self.masks[:num_of_imgs]

        assert len(self.masks) == len(self.images), "Mismatched number of masks and RGB images."

        # for i in range(len(self.images)):
        #     io.imsave("ts/rgb_{}.png".format(i + 1), np.uint8(self.images[i]))
        #     io.imsave("ts/mask_{}.png".format(i + 1), np.uint8(self.masks[i]))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):

        image = self.transform(self.images[idx])
        mask = self.masks[idx]
        label = self.labels[idx]

        return image, mask, label

    def __len__(self):
        return len(self.images)


class MaskTestset(Dataset):

    def __init__(self, filepath, num_of_imgs=0, resume_from=None):

        super(MaskTestset, self).__init__()

        self.filepath = filepath
        self.patch_size = patch_size

        if os.path.isdir(self.filepath):
            self.images_grid = []   # list ( n * 2 )
            self.imageIDX = []      # list ( n )
            self.image_size = []    # list ( ? )

            self.files = [f for f in sorted(os.listdir(self.filepath))
                          if os.path.isfile(os.path.join(self.filepath, f))]
            if resume_from is not None:
                self.files[:self.files.index(resume_from)] = []
            for i, file in enumerate(tqdm(self.files, desc="loading images")):
                if num_of_imgs != 0 and i == num_of_imgs:
                    break
                if file.endswith((".svs", ".tiff")):
                    self.mode = "WSI"
                    slide = OpenSlide(os.path.join(self.filepath, file))
                    patches_grid = self.sample_patches(slide.dimensions, self.patch_size - 16)
                    self.images_grid.extend(patches_grid)
                    self.imageIDX.extend([i] * len(patches_grid))
                    self.image_size.append(slide.dimensions)
                    slide.close()
                elif file.endswith((".jpg", ".png")):
                    self.mode = "ROI"
                    img = io.imread(os.path.join(self.filepath, file)).astype(np.uint8)
                    patches_grid = self.sample_patches(img.shape, self.patch_size - 16)
                    self.images_grid.extend(patches_grid)
                    self.imageIDX.extend([i] * len(patches_grid))
                    self.image_size.append(img.shape)
                else:
                    raise FileNotFoundError("Invalid data directory.")

        elif os.path.isfile(self.filepath) and self.filepath.endswith(("h5", "hdf5")):
            self.mode = "patch"
            self.images = []
            f = h5py.File(self.filepath, 'r')
            for i, img in enumerate(f['x']):
                if num_of_imgs != 0 and i == num_of_imgs:
                    break
                self.images.append(img)

        else:
            raise FileNotFoundError("Invalid data directory.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def sample_patches(self, size, interval):

        patches_grid = []
        xborder = size[0] - self.patch_size[0]
        yborder = size[1] - self.patch_size[1]

        if self.mode == "WSI":
            for x in np.arange(0, xborder + 1, interval[0]):
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((x, y))
                if patches_grid[-1][1] != yborder:
                    patches_grid.append((x, yborder))
                # return patches_grid

            if patches_grid[-1][0] != xborder:
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((xborder, y))

                if patches_grid[-1][1] != yborder:
                    patches_grid.append((xborder, yborder))

        elif self.mode == "ROI":
            for x in np.arange(0, xborder + 1, interval[0]):
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((x, y))
                if patches_grid[-1][1] != yborder:
                    patches_grid.append((x, yborder))

            if patches_grid[-1][0] != xborder:
                for y in np.arange(0, yborder + 1, interval[1]):
                    patches_grid.append((xborder, y))

                if patches_grid[-1][1] != yborder:
                    patches_grid.append((xborder, yborder))
        else:
            raise TypeError("Invalid image type.")
        return patches_grid

    def get_a_patch(self, idx):

        if self.mode == "WSI":
            image_file = os.path.join(self.filepath, self.files[self.imageIDX[idx]])
            slide = OpenSlide(image_file)
            x, y = self.images_grid[idx]
            patch = np.asarray(slide.read_region((x, y), level=0, size=tuple(self.patch_size)).convert('RGB'))
            slide.close()

        elif self.mode == "ROI":
            image_file = os.path.join(self.filepath, self.files[self.imageIDX[idx]])
            image = io.imread(image_file).astype(np.uint8)
            x, y = self.images_grid[idx]
            patch = image[x:x + self.patch_size[0], y:y + self.patch_size[1]]

        else:
            patch = self.images[idx]
        return patch, self.imageIDX[idx]

    def __getitem__(self, idx):

        patch, _ = self.get_a_patch(idx)
        patch = self.transform(patch)
        return patch

    def __len__(self):

        if self.mode == "patch":
            return len(self.images)
        else:
            return len(self.images_grid)


def get_tiles(image, interval, size):
    """
    划分实例。
    :param image: 输入图片矩阵，299 x 299 x 3
    :param interval: 取 tile 坐标点的间隔
    :param size: 单个 tile 的大小
    :return: 每个实例 tile 的左上角坐标的列表
    """

    tiles = []
    for x in np.arange(0, image.shape[0] - size + 1, interval):
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            tiles.append((x, y))

        if tiles[-1][1] + size != image.shape[1]:
            tiles.append((x, image.shape[1] - size))

    if tiles[-1][0] + size != image.shape[0]:
        for y in np.arange(0, image.shape[1] - size + 1, interval):
            tiles.append((image.shape[0] - size, y))

        if tiles[-1][1] + size != image.shape[1]:
            tiles.append((image.shape[0] - size, image.shape[1] - size))

    return tiles


def categorize(x):
    """按 LYSTO 划分的 7 个细胞数目类别划分分类标签。"""
    if x == 0:
        label = 0
    elif x <= 5:
        label = 1
    elif x <= 10:
        label = 2
    elif x <= 20:
        label = 3
    elif x <= 50:
        label = 4
    elif x <= 200:
        label = 5
    else:
        label = 6
    return label


def de_categorize(label):
    """给出每个 label 对应的范围。"""
    if label == 0:
        xmin, xmax = 0, 0
    elif label == 1:
        xmin, xmax = 1, 5
    elif label == 2:
        xmin, xmax = 6, 10
    elif label == 3:
        xmin, xmax = 11, 20
    elif label == 4:
        xmin, xmax = 21, 50
    elif label == 5:
        xmin, xmax = 51, 200
    else:
        xmin, xmax = 201, 100000
    return xmin, xmax


if __name__ == '__main__':

    batch_size = 2
    imageSet = LystoDataset("data/training.h5", tile_size=32, interval=150, num_of_imgs=51)
    imageSet_val = LystoDataset("data/training.h5", train=False, tile_size=32, interval=150, num_of_imgs=51)
    train_loader = DataLoader(imageSet, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(imageSet_val, batch_size=batch_size, shuffle=False)

    imageSet.setmode(1)
    imageSet_val.setmode(1)
    for idx, data in enumerate(train_loader):
        print('Dry run : [{}/{}]\r'.format(idx + 1, len(train_loader)))
    print("Length of dataset: {}".format(len(train_loader.dataset)))
    for idx, data in enumerate(val_loader):
        print('Dry run : [{}/{}]\r'.format(idx + 1, len(val_loader)))
    print("Length of dataset: {}".format(len(val_loader.dataset)))

    # 查看第一张图片
    print("The first training image: ")
    plt.imshow(imageSet.images[0])
    plt.show()
    print("Slide Patch: {0}\nLabel: {1}".format(imageSet.organs[0], imageSet.labels[0]))
    print("Grids of tiles: {}".format(imageSet.tiles_grid[0]))

    print("The first validation image: ")
    plt.imshow(imageSet_val.images[0])
    plt.show()
    print("Slide Patch: {0}\nLabel: {1}".format(imageSet_val.organs[0], imageSet_val.labels[0]))
    print("Grids of tiles: {}".format(imageSet_val.tiles_grid[0]))
