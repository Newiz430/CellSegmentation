import os
import glob
import configparser
import argparse
import time
import csv
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LystoTestset
from model import nets
from inference import inference_image

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_count.py", description='Cell count evaluation')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model')
parser.add_argument('-e', '--epoch', type=int, help='how many epochs the models have been trained')
parser.add_argument('-B', '--image_batch_size', type=int, default=64, help='batch size of images (default: 64)')
parser.add_argument('-c', '--cls_limit', action='store_true',
                    help='whether or not limiting counts by classification results')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0, help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now),
                    help='path of output details .csv file (default: ./output/<timestamp>)')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
args = parser.parse_args()


def test_ensemble(loader, models, epoch, cls_limit, output_path):

    fconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(now, epoch)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count', 'organ'])

    outputs = []
    for i, m in enumerate(models):
        print('Testing {}/{}...'.format(i + 1, len(models)))

        testset.setmode("image")
        outputs.append(inference_image(loader, m, device, mode='test', cls_limit=cls_limit)[1])

    # take average as the final result
    output = np.asarray(outputs).mean(axis=0)
    for i, y in enumerate(output, start=1):
        w.writerow([i, y, testset.organs[i - 1]])
    fconv.close()


if __name__ == "__main__":
    print("Testing settings: ")
    print("Device: {} | Model: {} | Image batch size: {} | Output directory: {}"
          .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.image_batch_size, args.output))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    testing_data_path = config.get("data", "data_path")
    # data loading
    testset = LystoTestset(os.path.join(testing_data_path, "test.h5"), num_of_imgs=20 if args.debug else 0)
    test_loader = DataLoader(testset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    models = []
    for m in glob.glob(os.path.join(args.model, '*_{}epochs.pth'.format(args.epoch))):

        f = torch.load(m, map_location=device)
        model = nets[f['encoder']]
        # load params of resnet encoder and image head only
        model.load_state_dict(
            OrderedDict({k: v for k, v in f['state_dict'].items()
                     if k.startswith(model.encoder_prefix + model.image_module_prefix)}),
            strict=False)
        model.setmode("image")
        model.to(device)
        models.append(model)

    test_ensemble(test_loader, models, args.epoch, args.cls_limit, output_path=args.output)
