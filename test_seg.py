import os
import argparse
import time
import csv
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MaskTestset
from model import encoders
from inference import inference_seg
from utils import save_images_with_masks

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_seg.py", description='Segmentation evaluation')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model')
parser.add_argument('--draw_masks', action='store_true', help='evaluation via computing binary masks')
parser.add_argument('--detect', action='store_true', help='evaluation via cell center localization')
parser.add_argument('-D', '--data_path', type=str, default='data/test.h5',
                    help='path to testing data (default: ./data/test.h5)')
parser.add_argument('-B', '--image_batch_size', type=int, default=64,
                    help='batch size of images (default: 64)')
parser.add_argument('-c', '--threshold', type=float, default=0.5,
                    help='minimal prob of pixels for generating segmentation masks (default: 0.5)')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now), metavar='OUTPUT/PATH',
                    help='path of output masked images (default: ./output/<timestamp>)')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
args = parser.parse_args()


def test_seg(testset, threshold, output_path):

    global epoch, model

    print('Start testing ...')
    masks = inference_seg(test_loader, model, device, mode='test')
    save_images_with_masks(testset.images, masks, threshold, output_path)


def test_detect(testset, output_path):

    # f1()
    pass


if __name__ == "__main__":

    print("Testing settings: ")
    print("Device: {} | Model: {} | Data directory: {} | Image batch size: {}\n"
          "Threshold: {} | Output directory: {}"
          .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.data_path,
                  args.image_batch_size, args.threshold, args.output))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    print('Loading Dataset ...')
    testset = MaskTestset(args.data_path, num_of_imgs=20 if args.debug else 0)
    test_loader = DataLoader(testset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    f = torch.load(args.model, map_location=device)
    model = encoders[f['encoder']]
    epoch = f['epoch']
    # load all params
    model.load_state_dict(
        OrderedDict({k: v for k, v in f['state_dict'].items()
                     if k.startswith(model.resnet_module_prefix + model.tile_module_prefix +
                                     model.image_module_prefix + model.seg_module_prefix)}),
        strict=False)
    model.setmode("segment")
    model.to(device)

    if args.draw_masks:
        test_seg(testset, args.threshold, output_path=args.output)
    elif args.detect:
        test_detect(testset, output_path=args.output)
    else:
        raise Exception("Something wrong in setting test modes. "
                        "Choose either \'--draw_masks\' or \'--detect\'. ")
