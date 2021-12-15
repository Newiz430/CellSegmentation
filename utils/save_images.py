import os
from tqdm import tqdm
import csv

import numpy as np
# from PIL import Image
import cv2
from skimage import io, morphology
from scipy import ndimage as ndi


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
        # Image.fromarray(np.uint8(img)).save(os.path.join(output_path, name))


def generate_masks(dataset, tiles, groups, save_masks=True, output_path="./data/pseudomask"):
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

    for i in tqdm(range(len(dataset.images)), desc="saving mask images"):
        preprocess(pseudo_masks[i])

        if save_masks:
            io.imsave(os.path.join(output_path, "rgb/{}.png".format(i + 1)), np.uint8(dataset.images[i]))
            io.imsave(os.path.join(output_path, "mask/{}.png".format(i + 1)), np.uint8(pseudo_masks[i] * 255))
            # Image.fromarray(np.uint8(dataset.images[i])).save(
            #     os.path.join(output_path, "rgb/{}.png".format(i + 1)),
            #     optimize=True)
            # Image.fromarray(np.uint8(pseudo_masks[i] * 255)).save(
            #     os.path.join(output_path, "mask/{}.png".format(i + 1)),
            #     optimize=True)

    if save_masks:
        print("Original images & masks saved in \'{}\'.".format(output_path))

    return pseudo_masks


def preprocess(mask):
    # remove small patches with low connectivity
    mask = morphology.erosion(mask, footprint=morphology.square(3))
    mask = morphology.erosion(mask, footprint=morphology.square(3))
    mask = morphology.remove_small_objects(mask, min_size=16*16+1)
    mask = ndi.binary_fill_holes(mask > 0)

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
#         Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(i)))


def heatmap(testset, tiles, probs, groups, csv_file, output_path):
    """
    tiles: up-left coordinates of tiles
    probs: positive probability of each tile
    groups: image indices for each tile
    """

    masks = np.zeros((len(testset.images), *testset.image_size)).astype(np.uint8)

    for i, g in enumerate(groups):
        tile_mask = np.full((testset.tile_size, testset.tile_size), probs[i])
        grid = list(map(int, tiles[i]))
        masks[g][grid[0]: grid[0] + testset.tile_size,
                 grid[1]: grid[1] + testset.tile_size] = tile_mask
        w = csv.writer(csv_file)
        w.writerow([g, '{}'.format(grid), probs[i]])

    for i, img in tqdm(enumerate(testset.images), desc="saving heatmaps"):

        mask = cv2.applyColorMap(255 - np.uint8(255 * masks[i]), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        io.imsave(os.path.join(output_path, "test_{}.png".format(groups[i] + 1)), np.uint8(img))

    # count = 0
    # # test_idx = len(testset)
    # test_idx = 20
    #
    # for i in range(1, len(groups) + 1):
    #     count += 1
    #
    #     if i == len(groups) or groups[i] != groups[i - 1]:
    #         img = testset.images[groups[i - 1]]
    #         mask = np.zeros(testset.image_size)
    #
    #         for j in range(i - count, i):
    #             tile_mask = np.full((testset.tile_size, testset.tile_size), probs[j])
    #             grid = list(map(int, tiles[j]))
    #
    #             mask[grid[0]: grid[0] + testset.tile_size,
    #                  grid[1]: grid[1] + testset.tile_size] = tile_mask
    #
    #             w = csv.writer(csv_file)
    #             w.writerow([groups[i - 1], '{}'.format(grid), probs[j]])
    #
    #         mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
    #         img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    #         io.imsave(os.path.join(output_path, "test_{}.png".format(groups[i - 1] + 1)), np.uint8(img))
    #         # Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(groups[i - 1] + 1)))
    #
    #         count = 0
    #
    #         # 没有阳性 tile 的时候。。。
    #         if i == len(groups) and groups[i - 1] != test_idx or groups[i - 1] != groups[i] - 1:
    #             for j in range(groups[i - 1] + 1, test_idx if i == len(groups) else groups[i]):
    #                 img = testset.images[j]
    #                 mask = np.zeros((img.shape[0], img.shape[1]))
    #                 mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
    #                 img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    #                 io.imsave(os.path.join(output_path, "test_{}.png".format(j + 1)), np.uint8(img))
    #                 # Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(j + 1)))


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
