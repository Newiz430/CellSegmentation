import os
from tqdm import tqdm
import csv

import numpy as np
import cv2
from skimage import io, morphology


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
        io.imsave(os.path.join(output_path, name), np.uint8(img))


def generate_masks(dataset, tiles, groups, preprocess, save_masks=True, output_path="./data/pseudomask"):
    """把预测得到的阳性细胞区域做成二值掩码。
    tiles: up-left coordinates of tiles
    groups: image indices for each tile
    """

    pseudo_masks = np.zeros((len(dataset.images), *dataset.image_size)).astype(np.uint8)

    for i in range(len(groups)):
        tile_mask = np.ones((dataset.tile_size, dataset.tile_size)).astype(np.uint8)
        grid = list(map(int, tiles[i]))

        pseudo_masks[groups[i]][grid[0]: grid[0] + dataset.tile_size,
                                grid[1]: grid[1] + dataset.tile_size] = tile_mask

    for i, img in enumerate(tqdm(dataset.images, desc="saving mask images")):
        if preprocess:
            pseudo_masks[i] = preprocess_masks(img, pseudo_masks[i])

        if save_masks:
            io.imsave(os.path.join(output_path, "rgb/{}.png".format(i + 1)), np.uint8(img))
            io.imsave(os.path.join(output_path, "mask/{}.png".format(i + 1)), np.uint8(pseudo_masks[i] * 255))

    if save_masks:
        print("Original images & masks saved in \'{}\'.".format(output_path))

    return pseudo_masks


def preprocess_masks(img, mask):

    # binarize original images by setting a value thresh in HSV
    img_split = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    _, mask_hsv = cv2.threshold(img_split[2], thresh=170, maxval=255, type=cv2.THRESH_BINARY)
    # intersect with mask for more precise borders
    mask = np.logical_and(mask, (1 - mask_hsv / 255).astype(bool))
    # remove small patches with low connectivity
    mask = morphology.remove_small_objects(mask, min_size=400)
    mask = morphology.remove_small_holes(mask, area_threshold=120)

    return mask


# def heatmap(testset, tiles, probs, topk, csv_file, output_path):
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
#         io.imsave(os.path.join(output_path, "test_{}.png".format(i)), np.uint8(img))


def heatmap(testset, tiles, probs, groups, csv_file, output_path):
    """
    tiles: up-left coordinates of tiles
    probs: positive probability of each tile
    groups: image indices for each tile
    """

    masks = np.zeros((len(testset.images), *testset.image_size))

    for i, g in enumerate(groups):
        tile_mask = np.full((testset.tile_size, testset.tile_size), probs[i])
        grid = list(map(int, tiles[i]))
        masks[g][grid[0]: grid[0] + testset.tile_size,
                 grid[1]: grid[1] + testset.tile_size] = tile_mask
        w = csv.writer(csv_file)
        w.writerow([g, '{}'.format(grid), probs[i]])

    for i, img in enumerate(tqdm(testset.images, desc="saving heatmaps")):

        mask = cv2.applyColorMap(255 - np.uint8(255 * masks[i]), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        io.imsave(os.path.join(output_path, "test_{}.png".format(i + 1)), np.uint8(img))


def save_images_with_masks(images, masks, threshold, output_path):
    """
    images: list of 3-d RGB np.ndarrays
    masks: list of 2-d output np.ndarrays
    """

    for i in tqdm(range(len(images)), desc="generating segmentation results"):
        mask = masks[i][0] > threshold

        for ch in range(3):
            images[i][:, :, ch] = images[i][:, :, ch] * 0.5 + np.uint8(255 * mask) * 0.5

        io.imsave(os.path.join(output_path, 'test_{}.png'.format(i + 1)), np.uint8(images[i]))

    print("Test results saved in \'{}\'.".format(output_path))


if __name__ == "__main__":
    from dataset import LystoDataset, LystoTestset

    # imageSet_test = LystoTestset("data/test.h5")
    # save_images(imageSet_test, 'test', './data/test')

    set = LystoDataset("data/training.h5")
    save_images(set, 'train', './data/train')
