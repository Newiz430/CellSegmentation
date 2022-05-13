import collections
import os
import traceback
import glob
import shutil
import argparse
import time
import csv
import simplejson
from collections import OrderedDict
from easydict import EasyDict as edict
from tqdm import tqdm

import numpy as np
import cv2
from skimage import io
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as tf

from dataset import get_tiles, MaskTestset, PointTestset
from model import nets
from inference import inference_seg
from metrics import dice_coef, euclid_dist, precision_recall
from utils import (dotting,
                   crop_wsi,
                   locate_cells,
                   sort_files,
                   overlap_mask,
                   remove_small_regions,
                   save_images_with_masks)

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_seg.py", description='Segmentation evaluation')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model')
parser.add_argument('--draw_masks', action='store_true', help='evaluation via computing binary masks')
parser.add_argument('--detect', action='store_true', help='evaluation via cell center localization')
parser.add_argument('--soft_mask', action='store_true', help='output soft masks to output_path/soft')
parser.add_argument('--test_qupath', action='store_true', help='test qupath predictions for comparison')
parser.add_argument('--area_type', action='store_true',
                    help='split test data by area type, conflict with --cancer_type')
parser.add_argument('--cancer_type', action='store_true',
                    help='split test data by cancer type, conflict with --area_type')
parser.add_argument('--smooth_method', type=str, default='gaussianblur',
                    help='smoothing method for cell detection, {\'gaussianblur\', \'distancetransform\'} '
                         '(default: \'gaussianblur\', no use with --draw_masks)')
parser.add_argument('--eps', type=float, default=11,
                    help='radius of DBSCAN in cell detection (default: 11, no use with --draw_masks)')
parser.add_argument('-r', '--reg_limit', action='store_true',
                    help='whether or not setting limitation on artifact patches by counting')
parser.add_argument('-D', '--data_path', type=str, default='./data/test.h5',
                    help='path to testing data (default: ./data/test.h5)')
parser.add_argument('-B', '--image_batch_size', type=int, default=128,
                    help='batch size of images (default: 128)')
parser.add_argument('-c', '--threshold', type=float, default=0.5,
                    help='minimal prob of pixels for generating segmentation masks '
                         '(default: 0.5, no use with --detect)')
parser.add_argument('-w', '--workers', type=int, default=4,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0)')
parser.add_argument('--save_image', action='store_true', help='save model prediction images')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now), metavar='OUTPUT/PATH',
                    help='path of output masked images (default: ./output/<timestamp>)')
parser.add_argument('--resume_from', type=str, default=None, metavar='IMAGE_FILE_NAME.<EXT>',
                    help='ROI image file name (path set in --data_path) to continue testing '
                         'if workers are killed halfway')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
args = parser.parse_args()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n

    @property
    def val(self):
        return self._val.item() if torch.is_tensor(self._val) else self._val

    @property
    def avg(self):
        avg = self._sum / self._count
        return avg.item() if torch.is_tensor(avg) else avg


class MetricGroup:
    def __init__(self):
        self.P = AverageMeter()
        self.R = AverageMeter()
        self.F1 = AverageMeter()
        self.dice = AverageMeter()

    def avg(self):
        return self.P.avg, self.R.avg, self.F1.avg, self.dice.avg

    def val(self):
        return self.P.val, self.R.val, self.F1.val, self.dice.val

    def update(self, vals):
        self.P.update(vals[0])
        self.R.update(vals[1])
        self.F1.update(vals[2])
        self.dice.update(vals[3])


def get_prf1(points_hat, points):
    CELL_RADIUS_PXS = 16

    # count true positives, false positives and false negatives
    flag = np.full(len(points), False)
    tp = 0
    for p_hat in points_hat:
        idx = None
        dmin = np.Inf
        for j, p in enumerate(points):
            if not flag[j]:
                dist = euclid_dist(p, p_hat)
                if dist < dmin:
                    idx = j
                    dmin = dist
        if dmin <= CELL_RADIUS_PXS:
            flag[idx] = True
            tp += 1
    fp = len(points_hat) - tp
    fn = sum(~flag)
    p, r, f1 = precision_recall(tp, fp, fn, return_f1=True)
    return p, r, f1, tp, fp, fn


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
                    classes = remove_small_regions(classes, min_object_size=300, hole_area_threshold=100)
                    soft_dir = os.path.join(output_path, 'soft')
                    if not os.path.exists(soft_dir):
                        os.makedirs(soft_dir)
                    io.imsave(os.path.join(soft_dir, '{:05}.png'.format(patch_id + 1)), np.uint8(255 * mask * classes))
                    mask = cv2.applyColorMap(255 - np.uint8(255 * mask * classes), cv2.COLORMAP_JET)

                    patch = cv2.addWeighted(patch, 0.5, mask, 0.5, 0)
                else:
                    overlap_mask(patch, classes)

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
                    output_grid, discarded = meanshift_cluster(whole_image_mask, method, int(cell_count),
                                                               eps=eps, **method_kwargs)
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
        output_grid, discarded = meanshift_cluster(whole_image_mask, method, int(cell_count),
                                                   eps=eps, **method_kwargs)
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


def meanshift_cluster(mask, method, cell_count=None, thr_for_setting_points=0.2, window_size=16, interval=10,
                      eps=15, **method_kwargs):
    """Meanshift clustering for excluding redundant points. Mask should be a 3-dimensional RGB image. """
    tiles = get_tiles(mask, interval, window_size)
    if method == "gaussianblur":
        mask = cv2.GaussianBlur(mask, **method_kwargs)
    elif method == "distancetransform":
        thr_for_dt = 10  # [0, 255]
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
    # crit_stop = (cv2.TERM_CRITERIA_COUNT, 5, 0)

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
    if cell_count is not None:
        return new_grids[:cell_count], new_grids[cell_count:]
    else:
        return new_grids, []


def test_qupath(mode="lysto", categorize_by=None):

    if not os.path.exists(os.path.join(args.data_path, 'qupath_centered')):
        os.makedirs(os.path.join(args.data_path, 'qupath_centered'))
    if not os.path.exists(os.path.join(args.data_path, 'qupath_predict_mask')):
        os.makedirs(os.path.join(args.data_path, 'qupath_predict_mask'))

    assert mode in {"lysto", "ihc"}
    filename_pattern = "(?<=test_)\d*" if mode == "lysto" else None
    testset = PointTestset(args.data_path, filename_pattern, num_of_imgs=1 if args.debug else 0)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers,
                             pin_memory=False)

    f = open(os.path.join(args.output, "center.csv"), 'w', newline="")
    w = csv.writer(f, delimiter=",")
    w.writerow(["id", "tp", "fp", "fn", "p", "r", "f1", "dice"])

    if mode == "lysto" and categorize_by == "cancer_type":
        metrics = {
            "breast": MetricGroup(),
            "colon": MetricGroup(),
            "prostate": MetricGroup()
        }
    elif categorize_by == "area_type":
        metrics = {
            "regular": MetricGroup(),
            "clustered": MetricGroup(),
            "artifact": MetricGroup()
        }
    else:
        metrics = MetricGroup()

    mask_files = sort_files(os.listdir(os.path.join(args.data_path, "qupath_mask")), filename_pattern)
    point_files = sort_files(os.listdir(os.path.join(args.data_path, "qupath_point")), filename_pattern)
    for i, (_, mask, points, cancer, area) in enumerate(tqdm(test_loader, desc="testing")):
        points_hat = []
        points = points[0]
        mask = mask[0].cpu().to(dtype=torch.float32)
        mask_hat = io.imread(os.path.join(args.data_path, "qupath_mask", mask_files[i]))
        mask_hat = mask_hat[..., 0] if mask_hat.ndim > 1 else mask_hat
        mask_hat = remove_small_regions(mask_hat, min_object_size=300, hole_area_threshold=100)
        dice = dice_coef(torch.from_numpy(mask_hat).to(dtype=torch.float32), mask / 255)
        # read points
        s = edict(simplejson.load(open(os.path.join(args.data_path, "qupath_point", point_files[i]), 'r')))
        for prop in s.features:
            if prop.properties.object_type == "detection" and prop.properties.classification.name == "Positive":
                # Approximately choose the first coordinate as the center point
                points_hat.append([max(0, np.round(prop.geometry.coordinates[0][0][0]).astype(int)),
                                   max(0, np.round(prop.geometry.coordinates[0][0][1]).astype(int))])
        p, r, f1, tp, fp, fn = get_prf1(np.asarray(points_hat), points)
        if mode == "lysto" and categorize_by == "cancer_type":
            metrics[cancer[0]].update([p, r, f1, dice])
        elif categorize_by == "area_type":
            metrics[area[0]].update([p, r, f1, dice])
        else:
            metrics.update([p, r, f1, dice])
        # draw labelled images
        original_img = testset.get_image(i).copy()
        color_gt = (255, 0, 0)  # RED
        color_hat = (0, 255, 0)  # GREEN
        dotting(original_img, points.cpu().numpy(), color=color_gt)
        dotting(original_img, np.asarray(points_hat), color=color_hat, thickness=2)
        io.imsave(os.path.join(args.data_path, 'qupath_centered', testset.names[i]), original_img)

        # cover masks on images
        original_img = testset.get_image(i).copy()
        overlap_mask(original_img, mask_hat, postprocess=False,
                     save=os.path.join(args.data_path, 'qupath_predict_mask', testset.names[i]))
        w.writerow([testset.names[i], str(tp), str(fp), str(fn), str(p), str(r), str(f1), str(dice)])

    if mode == "lysto" and categorize_by == "cancer_type":
        print("Breast: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['breast'].avg()))
        print("Colon: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['colon'].avg()))
        print("Prostate: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['prostate'].avg()))
    elif categorize_by == "area_type":
        print("Regular areas: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['regular'].avg()))
        print("Clustered cells: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['clustered'].avg()))
        print("Artifacts: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['artifact'].avg()))
    else:
        print("Average Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics.avg()))
        res = open("qupath_out.csv", 'a')
        resw = csv.writer(res, delimiter=',')
        resw.writerow(list(map(str, metrics.avg())))
        res.close()
    f.close()


def test(mode="lysto", categorize_by=None, method="gaussianblur", eps=15, save_image=True, **method_kwargs):

    if save_image:
        if not os.path.exists(os.path.join(args.data_path, 'centered')):
            os.makedirs(os.path.join(args.data_path, 'centered'))
        if not os.path.exists(os.path.join(args.data_path, 'predict_mask')):
            os.makedirs(os.path.join(args.data_path, 'predict_mask'))
        if not os.path.exists(os.path.join(args.data_path, 'predict_mask_binary')):
            os.makedirs(os.path.join(args.data_path, 'predict_mask_binary'))

    assert mode in {"lysto", "ihc"}
    filename_pattern = "(?<=test_)\d*" if mode == "lysto" else None
    testset = PointTestset(args.data_path, filename_pattern, num_of_imgs=1 if args.debug else 0)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers,
                             pin_memory=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)

    m = torch.load(args.model, map_location=device)
    model = nets[m['encoder']]
    # load all params
    model.load_state_dict(
        OrderedDict({k: v for k, v in m['state_dict'].items()
                     if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                     model.image_module_prefix + model.seg_module_prefix)}),
        strict=False)
    model.setmode("segment")
    model.to(device)
    model.eval()

    f = open(os.path.join(args.output, "center.csv"), 'w', newline="")
    w = csv.writer(f, delimiter=",")
    w.writerow(["id", "count", "tp", "fp", "fn", "p", "r", "f1", "dice"])

    if mode == "lysto" and categorize_by == "cancer_type":
        metrics = {
            "breast": MetricGroup(),
            "colon": MetricGroup(),
            "prostate": MetricGroup()
        }
    elif categorize_by == "area_type":
        metrics = {
            "regular": MetricGroup(),
            "clustered": MetricGroup(),
            "artifact": MetricGroup()
        }
    else:
        metrics = MetricGroup()

    with torch.no_grad():
        for i, (image, mask, points, cancer, area) in enumerate(tqdm(test_loader, desc="testing")):
            points = points[0]
            mask = mask[0].cpu().to(dtype=torch.float32)
            mask_hat = model(image.to(device)).to(dtype=torch.float32)
            mask_hat = F.softmax(mask_hat, dim=1)[:, 1][0].cpu().numpy()

            model.setmode("image")
            output_reg = model(image.to(device))[1].detach()[:, 0].clone().cpu()
            count = np.round(output_reg[0].item()).astype(int)
            model.setmode("segment")
            # cell count limitation
            if args.reg_limit and count == 0:
                mask_hat = 0 * mask_hat

            classes = mask_hat > args.threshold
            classes = remove_small_regions(classes, min_object_size=300, hole_area_threshold=100)
            dice = dice_coef(torch.from_numpy(classes).to(dtype=torch.float32), mask / 255)
            #
            # output_grid, discarded = meanshift_cluster(mask_hat * 255, method=method,
            #                                            cell_count=count if args.reg_limit else None,
            #                                            eps=eps, **method_kwargs)
            # output_grid = np.array([(y, x) for (x, y) in output_grid])
            # # if len(discarded) == 0:
            # #     count = len(output_grid)
            # p, r, f1, tp, fp, fn = get_prf1(output_grid, points)
            p, r, f1, tp, fp, fn = [0] * 6
            if mode == "lysto" and categorize_by == "cancer_type":
                metrics[cancer[0]].update([p, r, f1, dice])
            elif categorize_by == "area_type":
                metrics[area[0]].update([p, r, f1, dice])
            else:
                metrics.update([p, r, f1, dice])

            if save_image:
                # # draw labelled images
                # original_img = testset.get_image(i).copy()
                # color_gt = (255, 0, 0)  # RED
                # color_hat = (0, 255, 0)  # GREEN
                # dotting(original_img, points.cpu().numpy(), color=color_gt)
                # dotting(original_img, output_grid, color=color_hat, thickness=2)
                # io.imsave(os.path.join(args.data_path, 'centered', os.path.splitext(testset.names[i])[0]
                #                        + "_{}.png".format(str(count))), original_img)

                # cover masks on images
                original_img = testset.get_image(i).copy()
                io.imsave(os.path.join(args.data_path, 'predict_mask_binary', testset.names[i]), classes)
                overlap_mask(original_img, classes, postprocess=False,
                             save=os.path.join(args.data_path, 'predict_mask',
                                               os.path.splitext(testset.names[i])[0] + "_{}.png".format(str(count))))

            w.writerow([testset.names[i], str(count), str(tp), str(fp), str(fn),
                        str(p), str(r), str(f1), str(dice)])

    if mode == "lysto" and categorize_by == "cancer_type":
        print("Breast: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['breast'].avg()))
        print("Colon: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['colon'].avg()))
        print("Prostate: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['prostate'].avg()))
    elif categorize_by == "area_type":
        print("Regular areas: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['regular'].avg()))
        print("Clustered cells: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['clustered'].avg()))
        print("Artifacts: \nAverage Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics['artifact'].avg()))
    else:
        print("Average Precision: {}\nAverage Recall: {}\nAverage F1: {}\nAverage Dice: {}"
              .format(*metrics.avg()))
        res = open("out.csv", 'a')
        resw = csv.writer(res, delimiter=',')
        resw.writerow([str(args.threshold)] + list(map(str, metrics.avg())))
        res.close()

    if save_image:
        print("Test results saved in \'{}\' and \'{}\'.".format(os.path.join(args.data_path, 'centered'),
                                                                os.path.join(args.data_path, 'predict_mask')))

    f.close()


if __name__ == "__main__":

    print("Testing settings: ")
    print("Device: {} | Model: {} | Data directory: {} | Image batch size: {}"
          .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.data_path, args.image_batch_size))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.test_qupath:
        categorize_by = "area_type" if args.area_type else None
        print("Mode: {} (QuPath) | Categorize by: {}\nThreshold: {} | Smoothing method: {} | eps: {}"
              .format(os.path.basename(args.data_path),
                      categorize_by if categorize_by is not None else "",
                      args.threshold, args.smooth_method, args.eps))
        test_qupath(os.path.basename(args.data_path), categorize_by)
    elif args.draw_masks or args.detect:
        print("Mode: {} | {} | Output directory: {}"
              .format("segmentation" if args.draw_masks else "location detection",
                      "Threshold: {}".format(args.threshold) if args.draw_masks
                      else "Smoothing method: {} | eps: {}".format(args.smooth_method, args.eps),
                      args.output))
        if not args.data_path.endswith(("h5", "hdf5")):
            print("Cropping large WSIs to fit memory... ")
            crop_wsi(args.data_path)

        # data loading
        testset = MaskTestset(args.data_path, resume_from=args.resume_from, num_of_imgs=20 if args.debug else 0)
        test_loader = DataLoader(testset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=False)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
        m = torch.load(args.model, map_location=device)
        model = nets[m['encoder']]
        epoch = m['epoch']
        # load all params
        model.load_state_dict(
            OrderedDict({k: v for k, v in m['state_dict'].items()
                         if k.startswith(model.encoder_prefix + model.tile_module_prefix +
                                         model.image_module_prefix + model.seg_module_prefix)}),
            strict=False)
        model.setmode("segment")
        model.to(device)

        if args.draw_masks:
            test_seg(testset, args.threshold, soft=args.soft_mask, output_path=args.output)
        if args.detect:
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
        if args.area_type:
            categorize_by = "area_type"
        elif args.cancer_type:
            categorize_by = "cancer_type"
        else:
            categorize_by = None
        print("Mode: {} | Categorize by: {}\nThreshold: {} | Smoothing method: {} | eps: {}"
              .format(os.path.basename(args.data_path),
                      categorize_by if categorize_by is not None else "",
                      args.threshold, args.smooth_method, args.eps))
        test(os.path.basename(args.data_path), categorize_by=categorize_by, method=args.smooth_method,
             eps=args.eps, save_image=args.save_image, **smooth_params[args.smooth_method])

