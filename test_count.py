import os
import argparse
import time
import csv

import torch
from torch.utils.data import DataLoader

from dataset import LystoTestset
from model import encoders
from inference import inference_image

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_count.py", description='Cell count evaluation')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model')
parser.add_argument('-B', '--image_batch_size', type=int, default=64, help='batch size of images (default: 64)')
parser.add_argument('-c', '--cls_limit', action='store_true',
                    help='whether or not limiting counts by classification results')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0, help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now),
                    help='path of output details .csv file (default: ./output/<timestamp>)')
args = parser.parse_args()

print("Testing settings: ")
print("Device: {} | Model: {} | Image batch size: {} | Output directory: {}"
      .format('GPU' if torch.cuda.is_available() else 'CPU', args.model, args.image_batch_size, args.output))
if not os.path.exists(args.output):
    os.mkdir(args.output)

print('Loading Dataset ...')
imageSet_test = LystoTestset("data/test.h5")
test_loader = DataLoader(imageSet_test, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                         pin_memory=True)

f = torch.load(args.model)
model = encoders[f['encoder']]
epoch = f['epoch']
model.load_state_dict(f['state_dict'])

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
model.to(device)


def test_count(testset, cls_limit, output_path):

    global epoch, model

    fconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
        now, epoch)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count', 'organ'])

    print('Start testing ...')

    testset.setmode("count")
    model.setmode("image")
    output = inference_image(test_loader, model, device, mode='test', cls_limit=cls_limit)[1]
    # for i, y in enumerate(zip(*output), start=1):
    #     w.writerow([i, y[1], testset.organs[i - 1], y[0]])
    for i, y in enumerate(output, start=1):
        w.writerow([i, y, testset.organs[i - 1]])
    fconv.close()


def test_count_cls(testset, output_path):
    from inference import inference_image_cls
    global epoch, model

    fconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
        now, epoch)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count', 'organ'])

    print('Start testing ...')

    testset.setmode("count")
    model.setmode("image")
    output = inference_image_cls(test_loader, model, device, mode='test')
    # for i, y in enumerate(zip(*output), start=1):
    #     w.writerow([i, y[1], testset.organs[i - 1], y[0]])
    for i, y in enumerate(output, start=1):
        if y == 0:
            y = 0
        elif y == 1:
            y = 3
        elif y == 2:
            y = 8
        elif y == 3:
            y = 15
        elif y == 4:
            y = 30
        elif y == 5:
            y = 100
        w.writerow([i, y, testset.organs[i - 1]])
    fconv.close()


if __name__ == "__main__":

    # test_count(imageSet_test, args.cls_limit, output_path=args.output)
    test_count_cls(imageSet_test, output_path=args.output)
