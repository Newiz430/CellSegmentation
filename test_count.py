import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import LystoTestset
from model import encoders
from inference import inference_image

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_count.py", description='Cell count evaluation')
parser.add_argument('-m', '--model', type=str, default='checkpoint/checkpoint_10epochs.pth',
                    help='path to pretrained model (default: checkpoint/checkpoint_10epochs.pth)')
parser.add_argument('-B', '--image_batch_size', type=int, default=64, help='batch size of images (default: 64)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-d', '--device', type=int, default=0, help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now),
                    help='path of output details .csv file (default: ./output/<timestamp>)')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

print("Testing settings: ")
print("Model: {} | Image batch size: {} | Output directory: {}"
      .format(args.model, args.image_batch_size, args.output))
if not os.path.exists(args.output):
    os.mkdir(args.output)

print('Loading Dataset ...')
imageSet_test = LystoTestset("data/test.h5")
test_loader = DataLoader(imageSet_test, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                         pin_memory=False)

f = torch.load(args.model)
model = encoders[f['encoder']]
model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
epoch = f['epoch']
model.load_state_dict(f['state_dict'])

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
model.to(device)


def test_count(testset, output_path):

    global epoch, model

    fconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
        now, epoch)), 'w', newline="")
    w = csv.writer(fconv, delimiter=',')
    w.writerow(['id', 'count'])

    print('Start testing ...')

    # def predict_counts(loader, model, device):
    #     """预测测试集图片中阳性细胞的数目。"""
    #
    #     model.setmode("image")
    #     model.eval()
    #
    #     output = np.array([])
    #
    #     with torch.no_grad():
    #         image_bar = tqdm(loader, desc="cell counting")
    #         for input in image_bar:
    #             output = np.concatenate((output, model(input.to(device))[1].squeeze().cpu().numpy()))
    #
    #     # print("output.size = ", output.shape)
    #     return np.round(output).astype(int)

    testset.setmode("count")
    output = inference_image(test_loader, model, device, mode='test')[1]
    for i, count in enumerate(output, start=1):
        w.writerow([i, count])

    fconv.close()


if __name__ == "__main__":

    test_count(imageSet_test, output_path=args.output)
