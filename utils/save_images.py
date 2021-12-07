import os
from tqdm import tqdm
import csv

import numpy as np
from PIL import Image
import cv2
from skimage import io, morphology
from scipy import ndimage as ndi

import torch

def save_images(dataset, prefix, output_path, num_of_imgs=0):
    """把 hdf5 数据中的图像以 <name>_<idx>.png 的名称导出。

    :param dataset:     LystoDataset
    :param prefix:      所有图片共享的名称部分
    :param output_path: 输出图片路径
    :param num_of_imgs: 选择输出图片的数目（前 n 个）
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    bar = tqdm(dataset.images)
    for i, img in enumerate(bar):
        if num_of_imgs != 0 and i == num_of_imgs:
            break
        name = '{}_{}_{}cells.png'.format(prefix, i + 1, dataset.labels[i]) if hasattr(dataset, 'labels') \
            else '{}_{}.png'.format(prefix, i)
        Image.fromarray(np.uint8(img)).save(os.path.join(output_path, name))


def generate_masks(dataset, tiles, groups, save_masks=True, output_path="./data/pseudomask"):
    """把预测得到的阳性细胞区域做成二值掩码。"""

    pseudo_masks = np.zeros((len(dataset.images), *dataset.image_size)).astype(np.uint8)

    for i in range(len(groups)):

        tile_mask = np.ones((dataset.tile_size, dataset.tile_size)).astype(np.uint8)
        grid = list(map(int, tiles[i]))

        pseudo_masks[groups[i]][grid[0]: grid[0] + dataset.tile_size,
                                grid[1]: grid[1] + dataset.tile_size] = tile_mask

    for i in tqdm(range(len(dataset.images)), desc="saving mask images"):
        # remove small patches with low connectivity
        pseudo_masks[i] = morphology.remove_small_objects(pseudo_masks[i], min_size=32*32)
        pseudo_masks[i] = ndi.binary_fill_holes(pseudo_masks[i] > 0)

        if save_masks:
            Image.fromarray(np.uint8(dataset.images[i])).save(
                os.path.join(output_path, "rgb/{}.png".format(i + 1)),
                optimize=True)
            Image.fromarray(np.uint8(pseudo_masks[i] * 255)).save(
                os.path.join(output_path, "mask/{}.png".format(i + 1)),
                optimize=True)
    print("Original images & masks saved in {}.".format(output_path))

    return pseudo_masks


# def heatmap(testset, tiles, probs, topk, csv_file, output_path):
#     """把预测得到的阳性细胞区域标在图上。
#
#     :param testset:         测试集
#     :param tiles:           要标注的补丁
#     :param probs:           补丁对应的概率
#     :param topk:            标注的补丁数
#     :param output_path:     图像存储路径
#     """
#
#     for i, img in enumerate(testset.images):
#         mask = np.zeros((img.shape[0], img.shape[1]))
#         for idx in range(topk):
#             tile_mask = np.full((testset.size, testset.size), probs[idx + i * topk])
#             grid = list(map(int, tiles[idx + i * topk]))
#             mask[grid[0]: grid[0] + testset.size,
#                  grid[1]: grid[1] + testset.size] = tile_mask
#             # 输出信息
#             print("prob_{}:{}".format(i, probs[idx + i * topk]))
#             w = csv.writer(csv_file)
#             w.writerow([i, '{}'.format(grid), probs[idx + i * topk]])
#
#         mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
#         img = img * 0.5 + mask * 0.5
#         Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(i)))


def heatmap(testset, tiles, probs, groups, csv_file, output_path):
    """把预测得到的阳性细胞区域标在图上。"""

    global epoch

    count = 0
    # test_idx = len(testset)
    test_idx = 20

    for i in range(1, len(groups) + 1):
        count += 1

        if i == len(groups) or groups[i] != groups[i - 1]:
            img = testset.images[groups[i - 1]]
            mask = np.zeros(testset.image_size)

            for j in range(i - count, i):
                tile_mask = np.full((testset.tile_size, testset.tile_size), probs[j])
                grid = list(map(int, tiles[j]))

                mask[grid[0]: grid[0] + testset.tile_size,
                     grid[1]: grid[1] + testset.tile_size] = tile_mask

                # 输出信息
                # print("prob_{}:{}".format(groups[i - 1], probs[j]))
                w = csv.writer(csv_file)
                w.writerow([groups[i - 1], '{}'.format(grid), probs[j]])

            mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(groups[i - 1] + 1)))

            count = 0

            # 没有阳性 tile 的时候。。。
            if i == len(groups) and groups[i - 1] != test_idx or groups[i - 1] != groups[i] - 1:
                for j in range(groups[i - 1] + 1, test_idx if i == len(groups) else groups[i]):
                    img = testset.images[j]
                    mask = np.zeros((img.shape[0], img.shape[1]))
                    mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
                    img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
                    Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(j + 1)))


if __name__ == "__main__":
    from dataset import LystoDataset, LystoTestset

    # imageSet_test = LystoTestset("data/test.h5")
    # save_images(imageSet_test, 'test', './data/test')

    set = LystoDataset("data/training.h5")
    save_images(set, 'train', './data/train')
