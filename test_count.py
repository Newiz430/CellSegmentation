import os
import argparse
import configparser
import time
import csv
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LystoTestset
from model import nets
from inference import inference_image

now = int(time.time())
organs = ['colon', 'breast', 'prostate']

parser = argparse.ArgumentParser(prog="test_count.py", description='Cell count evaluation')
parser.add_argument('-m', '--model', type=str, nargs='+',
                    help='path to pretrained model '
                         '(3 models in order \'colon, breast, prostate\' if models for each organ are specified)')
parser.add_argument('-B', '--image_batch_size', type=int, default=64, help='batch size of images (default: 64)')
parser.add_argument('-c', '--cls_limit', action='store_true',
                    help='whether or not limiting counts by classification results')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0, help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now),
                    help='path of output details .csv file (default: ./output/<timestamp>)')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
args = parser.parse_args()


def test_count(loader, model, epoch, cls_limit, output_path):

    fconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
        now, epoch)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count', 'organ'])

    print('Start testing ...')

    testset.setmode("image")
    counts = inference_image(loader, model, device, mode='test', cls_limit=cls_limit)[1]
    for i, y in enumerate(counts, start=1):
        w.writerow([i, y, testset.organs[i - 1]])
    fconv.close()


def test_count_by_organ(loaders, models, cls_limit, output_path):

    fconv = open(os.path.join(output_path, '{}-count.csv'.format(now)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count'])

    print('Start testing ...')

    colonset.setmode("image")
    breastset.setmode("image")
    prostateset.setmode("image")
    ids = np.array([])
    counts = np.array([])
    for i in range(3):
        organ_ids, _, organ_counts = inference_image(loaders[i], models[i], device, mode='test', cls_limit=cls_limit,
                                                     return_id=True)
        for j in range(len(organ_ids)):
            print("id: {}, organ: {}, pred: {}".format(int(organ_ids[j]), organs[i], int(organ_counts[j])))
        ids = np.concatenate((ids, organ_ids))
        counts = np.concatenate((counts, organ_counts))

    counts = counts[np.argsort(ids.astype(int))]
    for i, y in enumerate(counts, start=1):
        w.writerow([i, y])
    fconv.close()


def load_model(path, device):
    f = torch.load(path, map_location=device)
    model = nets[f['encoder']]
    epoch = f['epoch']
    # load params of resnet encoder and image head only
    model.load_state_dict(
        OrderedDict({k: v for k, v in f['state_dict'].items()
                     if k.startswith(model.encoder_prefix + model.image_module_prefix)}),
        strict=False)
    model.setmode("image")
    model.to(device)
    return model, epoch


if __name__ == "__main__":
    print("Testing settings: ")
    print("Device: {} | Model: {} | Image batch size: {} | Output directory: {}"
          .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.image_batch_size, args.output))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # data loading
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    testing_data_path = config.get("data", "data_path")

    if len(args.model) == 3:
        colonset = LystoTestset(os.path.join(testing_data_path, "test.h5"), organ='colon', num_of_imgs=20 if args.debug else 0)
        breastset = LystoTestset(os.path.join(testing_data_path, "test.h5"), organ='breast', num_of_imgs=20 if args.debug else 0)
        prostateset = LystoTestset(os.path.join(testing_data_path, "test.h5"), organ='prostate', num_of_imgs=20 if args.debug else 0)

        colonloader = DataLoader(colonset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)
        breastloader = DataLoader(breastset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)
        prostateloader = DataLoader(prostateset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                                    pin_memory=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
        models = []
        for p in args.model:
            model, epoch = load_model(p, device)
            models.append(model)

        test_count_by_organ([colonloader, breastloader, prostateloader], models, args.cls_limit, output_path=args.output)

    elif len(args.model) == 1:
        testset = LystoTestset(os.path.join(testing_data_path, "test.h5"), num_of_imgs=20 if args.debug else 0)
        test_loader = DataLoader(testset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
        model, epoch = load_model(args.model[0], device)

        test_count(test_loader, model, epoch, args.cls_limit, output_path=args.output)

    else:
        raise Exception("Expected the number of pretrained models to be 1 or 3. ")
