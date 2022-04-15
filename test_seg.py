import os
import glob
import shutil
import argparse
import time
import csv
from collections import OrderedDict
import traceback
from tqdm import tqdm

import numpy as np
import cv2
from skimage import io
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_tiles, MaskTestset
from model import nets
from inference import inference_seg
from utils import save_images_with_masks

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_seg.py", description='Segmentation evaluation')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model')
parser.add_argument('--draw_masks', action='store_true', help='evaluation via computing binary masks')
parser.add_argument('--detect', action='store_true', help='evaluation via cell center localization')
parser.add_argument('--soft_mask', action='store_true', help='output soft masks to output_path/soft')
parser.add_argument('--smooth_method', type=str, default='gaussianblur',
                    help='smoothing method for cell detection, {\'gaussianblur\', \'distancetransform\'} '
                         '(default: \'gaussianblur\', using with --detect)')
parser.add_argument('-e', '--eps', type=int, default=15,
                    help='radius of DBSCAN in cell detection (default: 15, using with --detect)')
parser.add_argument('-D', '--data_path', type=str, default='./data/test.h5',
                    help='path to testing data (default: ./data/test.h5)')
parser.add_argument('-B', '--image_batch_size', type=int, default=128,
                    help='batch size of images (default: 128)')
parser.add_argument('-c', '--threshold', type=float, default=0.5,
                    help='minimal prob of pixels for generating segmentation masks '
                         '(default: 0.5, using with --draw_masks)')
parser.add_argument('-w', '--workers', type=int, default=4,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now), metavar='OUTPUT/PATH',
                    help='path of output masked images (default: ./output/<timestamp>)')
parser.add_argument('--resume_from', type=str, default=None, metavar='IMAGE_FILE_NAME.<EXT>',
                    help='ROI image file name (path set in --data_path) to continue testing '
                         'if workers are killed halfway')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
args = parser.parse_args()


def test_seg(testset, threshold, soft=False, output_path=None):

    global epoch, model

    model.eval()

    if testset.mode == "patch":
        masks = inference_seg(test_loader, model, device, mode='test')
        save_images_with_masks(testset.images, masks, threshold, output_path, soft=soft)

    else:
        seg_bar = tqdm(test_loader, desc="segmenting & mask generating")
        for b, images in enumerate(seg_bar):
            with torch.no_grad():
                output = model(images.to(device))
                output = F.softmax(output, dim=1)[:, 1].cpu().numpy()  # note: channel 1 for pos_mask=1 and bg=0

            for i, mask in enumerate(output):
                patch_id = b * len(output) + i
                patch, slideidx = testset.get_a_patch(patch_id)
                classes = mask > threshold

                if soft:
                    soft_dir = os.path.join(output_path, 'soft')
                    if not os.path.exists(soft_dir):
                        os.makedirs(soft_dir)
                    io.imsave(os.path.join(soft_dir, '{:05}.png'.format(patch_id + 1)), np.uint8(255 * mask * classes))
                    mask = cv2.applyColorMap(255 - np.uint8(255 * mask * classes), cv2.COLORMAP_JET)

                    patch = cv2.addWeighted(patch, 0.5, mask, 0.5, 0)
                else:
                    for ch in range(3):
                        patch[:, :, ch] = patch[:, :, ch] * 0.5 + np.uint8(255 * classes) * 0.5

                io.imsave(os.path.join(output_path, 'test_{:05}.png'.format(patch_id + 1)), np.uint8(patch))

        print("Test results saved in \'{}\'.".format(output_path))


def cell_detect(testset, resume=False, output_image=True, output_path=None, method="gaussianblur", eps=15,
                **method_kwargs):

    global epoch, model

    detect_path = os.path.join(output_path, 'detect')
    tmp_path = os.path.join(detect_path, 'tmp.csv')
    if not os.path.exists(detect_path):
        os.makedirs(detect_path)
    if resume:
        fpath = glob.glob(os.path.join(detect_path, '*-location.csv'))[-1]
        f = open(fpath, 'a', newline="")
        w = csv.writer(f, delimiter=',')
    else:
        fpath = os.path.join(detect_path, '{}-location.csv'.format(now))
        f = open(fpath, 'w', newline="")
        w = csv.writer(f, delimiter=',')
        w.writerow(['image_id', 'x', 'y'])

    print('Start testing ...')
    model.eval()

    imageIDX = None
    image_file = None
    whole_image_mask = None
    cell_count = None

    try:

        seg_bar = tqdm(test_loader, desc="segmenting & mask generating")
        # 按 batch 前向计算
        for b, images in enumerate(seg_bar):
            batch_counts = np.array([])
            with torch.no_grad():
                output = model(images.to(device))
                output = F.softmax(output, dim=1)[:, 1].cpu().numpy()  # note: channel 1 for pos_mask=1 and bg=0
                model.setmode("image")
                output_reg = model(images.to(device))[1].detach()[:, 0].clone().cpu()
                output_reg = np.round(output_reg.numpy()).astype(int)
                batch_counts = np.concatenate((batch_counts, output_reg))
                model.setmode("segment")

            # 把每个 batch 中的 mask 拿出来操作
            for i, mask in enumerate(output):
                mask = np.uint8(255 * mask)  # no threshold
                patch_id = b * len(output) + i  # mask 的 patch 索引，用于查找对应的 slide
                slideidx = testset.imageIDX[patch_id]  # slideidx 表示 mask 属于第几张图
                mask_grid = testset.images_grid[patch_id]  # mask_grid 表示 mask 在 slide 上的位置(左上角的坐标)
                # 判断当前 imageIDX，如果还是这一张，就继续填充，不是，就进行聚类、检测、输出、释放内存，新建下一张图
                if imageIDX is None:
                    imageIDX = slideidx
                    image_file = os.path.splitext(testset.files[imageIDX])[0]
                    whole_image_mask = np.zeros(testset.image_size[imageIDX][:-1], dtype=np.uint8)
                    cell_count = 0

                elif slideidx != imageIDX:
                    io.imsave(os.path.join(detect_path, 'mask_{}.png'.format(image_file)), whole_image_mask)
                    print("total number of cells in image \'{}\' : {}".format(image_file, cell_count))
                    output_grid, discarded = meanshift_cluster(whole_image_mask, method, int(cell_count), eps=eps,
                                                               **method_kwargs)
                    for x, y in output_grid:
                        if image_file.find("-") > 0:
                            x += int(image_file.split(sep='-', maxsplit=1)[1])
                            w.writerow([image_file.split(sep='-', maxsplit=1)[0], x, y])
                        else:
                            w.writerow([image_file, x, y])

                    if output_image:
                        output_slide = locate_cells(testset, imageIDX, output_grid, discarded)
                        io.imsave(os.path.join(detect_path, '{}_{}cells.png'.format(image_file, int(cell_count))),
                                  output_slide)

                    imageIDX = slideidx
                    image_file = os.path.splitext(testset.files[imageIDX])[0]
                    whole_image_mask = np.zeros(testset.image_size[imageIDX][:-1], dtype=np.uint8)
                    cell_count = 0

                whole_image_mask[mask_grid[0]:mask_grid[0] + testset.patch_size[0],
                                 mask_grid[1]:mask_grid[1] + testset.patch_size[1]] = mask
                cell_count += batch_counts[i]

        io.imsave(os.path.join(detect_path, 'mask_{}.png'.format(image_file)), whole_image_mask)
        print("total number of cells in image \'{}\' : {}".format(image_file, cell_count))
        output_grid, discarded = meanshift_cluster(whole_image_mask, method, int(cell_count), eps=eps, **method_kwargs)
        for x, y in output_grid:
            if image_file.find("-") > 0:
                x += int(image_file.split(sep='-', maxsplit=1)[1])
            w.writerow([image_file, x, y])
        if output_image:
            output_slide = locate_cells(testset, imageIDX, output_grid, discarded)
            io.imsave(os.path.join(detect_path, '{}_{}cells.png'.format(image_file, int(cell_count))), output_slide)

        f.close()

        # format correction
        tmp = open(tmp_path, 'w', newline="")
        f = open(fpath, 'r')
        r = csv.reader(f, delimiter=',')
        w = csv.writer(tmp, delimiter=',')
        for row in r:
            row[0] = row[0].split(sep='-', maxsplit=1)[0]
            w.writerow(row)
        f.close()
        tmp.close()
        os.remove(fpath)
        os.rename(tmp_path, fpath)

    except RuntimeError:

        del output, output_grid, output_slide, whole_image_mask
        f.close()

        # rollback
        f = open(fpath, 'r')
        tmp = open(tmp_path, 'w', newline="")
        r = csv.reader(f, delimiter=',')
        w = csv.writer(tmp, delimiter=',')
        for row in r:
            if not row[0] == image_file:
                w.writerow(row)
        f.close()
        tmp.close()
        os.remove(fpath)
        os.rename(tmp_path, fpath)

        traceback.print_exc()
        print("Exception catched! Current results saved in \'{}\'.\n"
              "If workers are killed unexpectedly by cache overflow, "
              "you may run this script again with extra argument \'--resume_from {}.png\'?\n"
              "See \'python test_seg.py -h\' for more details. "
              .format(fpath, image_file))

    finally:
        print("Test results saved in \'{}\'.".format(detect_path))


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
                    io.imsave(os.path.join(data_path, "{}-{}.png".format(file, border[i])), wsi[:, border[i]:border[i + 1]])

                shutil.move(os.path.join(data_path, file + '.png'), backup_path)

    for file in sorted(os.listdir(data_path)):
        if os.path.getsize(os.path.join(data_path, file)) > max_size:
            print("Huge images still exist. Cropping again... ")
            crop_wsi(data_path)


def meanshift_cluster(mask, method, cell_count, thr_for_setting_points=0.2, window_size=16, interval=10, eps=15,
                      **method_kwargs):
    """Meanshift clustering for excluding redundant points. """
    tiles = get_tiles(mask, interval, window_size)
    if method == "gaussianblur":
        mask = cv2.GaussianBlur(mask, **method_kwargs)
    elif method == "distancetransform":
        thr_for_dt = 120  # [0, 255]
        mask = cv2.distanceTransform(np.asarray((mask > thr_for_dt) * 255, dtype=np.uint8), **method_kwargs)
        mask = cv2.normalize(mask, mask, 0, 1, cv2.NORM_MINMAX) * 255
        mask = np.round(mask).astype(np.uint8)
    else:
        raise Exception("Smoothing method not found. ")

    # (width(y), height(x)) in real image, (x, y) in code
    track_windows = [(y, x, window_size, window_size) for (x, y) in tiles if
                     mask[x + window_size // 2, y + window_size // 2] > thr_for_setting_points * 255]

    grids = []
    iters = []  # number of iters, for debugging
    crit_stop = (cv2.TERM_CRITERIA_EPS, 0, 0.00001)

    for tw in tqdm(track_windows, desc="cell clustering", colour='red', leave=False):
        ret, (y, x, w, h) = cv2.meanShift(mask, tw, crit_stop)
        grids.append((x + window_size // 2, y + window_size // 2))
        iters.append(ret)

    grids = np.asarray(grids)
    new_grids = []

    if len(grids) != 0:
        grid_labels = DBSCAN(eps, min_samples=1, n_jobs=-1).fit_predict(grids)
        for i in range(np.max(grid_labels) + 1):
            idx = np.column_stack((grid_labels == i, grid_labels == i))
            new_grids.append(np.mean(grids, axis=0, where=idx).round().astype(int))

        # match grids with cell assessment by max pooling
        grid_weight = np.asarray([mask[x, y] for x, y in new_grids])
        new_grids = np.asarray(new_grids)[np.argsort(grid_weight)[::-1]]

    # if number of cells larger than points, do nothing
    # otherwise set a limit for points
    return new_grids[:cell_count], new_grids[cell_count:]


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


if __name__ == "__main__":

    print("Testing settings: ")
    print("Device: {} | Model: {} | Data directory: {} | Image batch size: {}\n"
          "Mode: {} | {} | Output directory: {}"
          .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.data_path, args.image_batch_size,
                  "segmentation" if args.draw_masks else "location detection",
                  "Threshold: {}".format(args.threshold) if args.draw_masks
                      else "Smoothing method: {}".format(args.smooth_method),
                  args.output))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("Cropping large WSIs to fit memory... ")
    crop_wsi(args.data_path)

    # data loading
    testset = MaskTestset(args.data_path, resume_from=args.resume_from, num_of_imgs=1 if args.debug else 0)
    # testset = MaskTestset(args.data_path, resume_from=args.resume_from, num_of_imgs=20 if args.debug else 0)
    test_loader = DataLoader(testset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    f = torch.load(args.model, map_location=device)
    model = nets[f['encoder']]
    epoch = f['epoch']
    # load all params
    model.load_state_dict(
        OrderedDict({k: v for k, v in f['state_dict'].items()
                     if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                     model.image_module_prefix + model.seg_module_prefix)}),
        strict=False)
    model.setmode("segment")
    model.to(device)

    if args.draw_masks:
        test_seg(testset, args.threshold, soft=args.soft_mask, output_path=args.output)
    elif args.detect:
        smooth_params = {
            "gaussianblur": {
                "ksize": (15, 15),
                "sigmaX": 3.
            },
            "distancetransform": {
                "distanceType": cv2.DIST_L2,
                "maskSize": cv2.DIST_MASK_PRECISE
            }
        }
        cell_detect(testset, resume=args.resume_from is not None, output_image=True, output_path=args.output,
                    method=args.smooth_method, eps=args.eps, **smooth_params[args.smooth_method])
    else:
        raise Exception("Something wrong in setting test modes. "
                        "Choose either \'--draw_masks\' or \'--detect\'. "
                        "See \'python test_seg.py -h\' for more details. ")
