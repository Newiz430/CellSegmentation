import warnings
import os
from tqdm import tqdm
import csv
import shutil

import numpy as np
import cv2
from skimage import io, morphology

warnings.filterwarnings("ignore")


def remove_small_regions(img_bin, min_object_size, hole_area_threshold):
    img_bin = morphology.remove_small_objects(img_bin, min_size=min_object_size)
    img_bin = morphology.remove_small_holes(img_bin, area_threshold=hole_area_threshold)
    return img_bin


def overlap_mask(img, img_bin, postprocess=True, min_object_size=300, hole_area_threshold=100, save=None):
    if postprocess:
        img_bin = remove_small_regions(img_bin, min_object_size=min_object_size,
                                       hole_area_threshold=hole_area_threshold)
    for i in range(3):
        img[:, :, i] = img[:, :, i] * 0.5 + np.uint8(255 * img_bin) * 0.5  # half black image + half white mask
    if save is not None:
        io.imsave(save, img)
    return img


def dotting(img, points, radius=4, color=(255, 0, 0), thickness=cv2.FILLED, **kwargs):
    for x, y in points:
        img = cv2.circle(img, (x, y), radius, color, thickness, **kwargs)
    return img


def locate_cells(dataset, slideidx, grids, discarded_grids=None):

    color = [255, 0, 0]
    slide_file = dataset.files[slideidx]
    slide = io.imread(os.path.join(dataset.filepath, slide_file)).astype(np.uint8)
    for y, x in grids:
        slide = cv2.circle(slide, (x, y), 4, color, cv2.FILLED)
    if discarded_grids is not None:
        color_discarded = [0, 0, 255]
        for y, x in discarded_grids:
            slide = cv2.circle(slide, (x, y), 4, color_discarded, cv2.FILLED)

    return slide


def save_images(dataset, prefix, output_path, num_of_imgs=0):
    """Export images in hdf5 data as '<name>_<idx>.png'.

    :param dataset:     LystoDataset
    :param prefix:      shared prefix of all images
    :param output_path: output directory
    :param num_of_imgs: number of output images (first n images)
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    f = open(os.path.join(output_path, '{}_organ.csv'.format(prefix)), 'w', newline="")
    w = csv.writer(f, delimiter=",")

    bar = tqdm(zip(dataset.images, dataset.organs))
    for i, (img, org) in enumerate(bar):
        if num_of_imgs != 0 and i == num_of_imgs:
            break
        name = '{}_{}_{}cells.png'.format(prefix, i + 1, dataset.labels[i]) if hasattr(dataset, 'labels') \
            else '{}_{}.png'.format(prefix, i + 1)
        # io.imsave(os.path.join(output_path, name), np.uint8(img))
        w.writerow([name, dataset.labels[i], org] if hasattr(dataset, 'labels') else [name, org])

    f.close()


def generate_masks(dataset, tiles, groups, preprocess, save_masks=True, output_path="./data/pseudomask"):
    """Transform predicted pos cell regions into binary masks.

    tiles: up-left coordinates of tiles
    groups: image indices for each tile
    """

    if not os.path.exists(os.path.join(output_path, "rgb")):
        os.makedirs(os.path.join(output_path, "rgb"))
    if not os.path.exists(os.path.join(output_path, "mask")):
        os.makedirs(os.path.join(output_path, "mask"))

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
            io.imsave(os.path.join(output_path, "rgb/{:05}.png".format(i + 1)), np.uint8(img))
            io.imsave(os.path.join(output_path, "mask/{:05}.png".format(i + 1)), np.uint8(pseudo_masks[i] * 255))

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
    mask = remove_small_regions(mask, min_object_size=400, hole_area_threshold=120)

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
#             # output info
#             print("prob_{}:{}".format(i, probs[idx + i * topk]))
#             w = csv.writer(csv_file)
#             w.writerow([i, '{}'.format(grid), probs[idx + i * topk]])
#
#         mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
#         img = img * 0.5 + mask * 0.5
#         io.imsave(os.path.join(output_path, "test_{:05}.png".format(i)), np.uint8(img))


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
        io.imsave(os.path.join(output_path, "test_{:05}.png".format(i + 1)), np.uint8(img))


def save_images_with_masks(images, masks, threshold, output_path, soft=False):
    """
    images: list of 3-d RGB np.ndarrays (3, h, w)
    masks: list of 2-d output np.ndarrays (h, w)
    """

    for i in tqdm(range(len(images)), desc="generating segmentation results"):
        classes = masks[i] > threshold
        if soft:
            soft_dir = os.path.join(output_path, 'soft')
            if not os.path.exists(soft_dir):
                os.makedirs(soft_dir)
            io.imsave(os.path.join(soft_dir, '{:05}.png'.format(i + 1)), np.uint8(255 * masks[i] * classes))
            mask = cv2.applyColorMap(255 - np.uint8(255 * masks[i] * classes), cv2.COLORMAP_JET)
            images[i] = cv2.addWeighted(images[i], 0.5, mask, 0.5, 0)
        else:
            for ch in range(3):
                images[i][:, :, ch] = images[i][:, :, ch] * 0.5 + np.uint8(255 * classes) * 0.5

        io.imsave(os.path.join(output_path, 'test_{:05}.png'.format(i + 1)), np.uint8(images[i]))

    print("Test results saved in \'{}\'.".format(output_path))


def crop_wsi(data_path, max_size=5e+7):

    backup_path = os.path.join(data_path, 'backup')
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)

    for file in tqdm(sorted(os.listdir(data_path)), desc="WSI size checking & cropping"):
        if os.path.getsize(os.path.join(data_path, file)) > max_size:
            wsi = io.imread(os.path.join(data_path, file))
            file = os.path.splitext(file)[0]
            if file.find("-") > 0:
                xorigin = int(file.split(sep='-', maxsplit=1)[1])
                border = np.linspace(xorigin, xorigin + wsi.shape[1], 2 + 1, dtype=int)
                for i in range(2):
                    io.imsave(os.path.join(data_path,
                                           "{}-{}.png".format(file.split(sep='-', maxsplit=1)[0], border[i])),
                              wsi[:, border[i] - xorigin:border[i + 1] - xorigin])
            else:
                border = np.linspace(0, wsi.shape[1], 5 + 1, dtype=int)
                for i in range(5):
                    io.imsave(os.path.join(data_path, "{}-{}.png".format(file, border[i])),
                              wsi[:, border[i]:border[i + 1]])

                shutil.move(os.path.join(data_path, file + '.png'), backup_path)

    for file in sorted(os.listdir(data_path)):
        if os.path.getsize(os.path.join(data_path, file)) > max_size:
            print("Huge images still exist. Cropping again... ")
            crop_wsi(data_path)


if __name__ == "__main__":
    from dataset import LystoDataset, LystoTestset

    imageset = LystoTestset("../data/test.h5")
    save_images(imageset, 'test', '../data/test')
    # imageset = LystoDataset("data/training.h5")
    # save_images(imageset, 'train', './data/train')
