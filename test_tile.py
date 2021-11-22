import os
import numpy as np
import argparse
import time
from tqdm import tqdm
import csv
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import encoders
from inference import inference_tiles
from save_images import heatmap

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_tile.py", description='Testing & Heatmap')
parser.add_argument('-m', '--model', type=str, default='checkpoint/checkpoint_10epochs.pth',
                    help='path to pretrained model (default: checkpoint/checkpoint_10epochs.pth)')
parser.add_argument('-b', '--batch_size', type=int, default=40960,
                    help='batch size of tiles (default: 40960)')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
# parser.add_argument('-k', '--topk', default=30, type=int,
#                     help='top k tiles are assumed to be of the same class as the image (default: 10, standard MIL)')
parser.add_argument('-i', '--interval', type=int, default=5,
                    help='sample interval of tiles (default: 5)')
parser.add_argument('-p', '--tile_size', type=int, default=32,
                    help='size of each tile (default: 32)')
parser.add_argument('-c', '--threshold', type=float, default=0.88,
                    help='minimal prob for tiles to show in heatmap (default: 0.88)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0)')
parser.add_argument('-o', '--output', type=str, default='output/{}'.format(now), metavar='OUTPUT/PATH',
                    help='path of output details .csv file (default: ./output/<timestamp>)')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

print("Testing settings: ")
print("Model: {} | Tiles batch size: {} | Tile size: {} | Interval: {} | Threshold: {} | Output directory: {}"
      .format(args.model, args.batch_size, args.tile_size, args.interval, args.threshold, args.output))
if not os.path.exists(args.output):
    os.mkdir(args.output)

f = torch.load(args.model)
model = encoders[f['encoder']]
model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
epoch = f['epoch']
model.load_state_dict(f['state_dict'])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.ToTensor(), normalize])

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
model.to(device)


def test_tile(testset, batch_size, workers, output_path):
    """
    :param testset:         测试数据集
    :param batch_size:      Dataloader 打包的小 batch 大小
    :param workers:         Dataloader 使用的进程数
    :param model:           网络模型
    :param output_path:     保存模型文件的目录
    """

    global epoch, model

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    # 热图中各个 tile 的信息保存在 output_path/<timestamp>-pred-<epoch>-<tilesize>-<interval>-<threshold>.csv
    fconv = open(os.path.join(output_path, '{}-pred-e{}-p{}-i{}-c{}.csv'.format(
        now, epoch, args.tile_size, args.interval, args.threshold)), 'w', newline="")
    w = csv.writer(fconv)
    w.writerow(['tile_size', '{}'.format(testset.tile_size)])
    w.writerow(['interval', '{}'.format(testset.interval)])
    w.writerow(['idx', 'grid', 'prob'])
    fconv.close()

    def rank(testset, probs):
        """按概率对 tile 排序，便于与置信度进行比较。"""

        groups = np.array(testset.tileIDX)
        tiles = np.array(testset.tiles_grid)

        order = np.lexsort((probs, groups))
        groups = groups[order]
        probs = probs[order]
        tiles = tiles[order]

        # index = np.empty(len(groups), 'bool')
        # index[-topk:] = True
        # index[:-topk] = groups[topk:] != groups[:-topk]
        index = [prob > args.threshold for prob in probs]

        return tiles[index], probs[index], groups[index]

    print('Start testing ...')

    testset.setmode("tile")
    probs = inference_tiles(test_loader, model, device, mode='test')
    tiles, probs, groups = rank(testset, probs)

    # 生成热图
    fconv = open(os.path.join(output_path, '{}-pred-e{}-p{}-i{}-c{}.csv'.format(
        now, epoch, args.tile_size, args.interval, args.threshold)), 'a', newline="")
    heatmap(testset, tiles, probs, groups, fconv, output_path)
    fconv.close()


if __name__ == "__main__":
    from dataset.dataset import LystoTestset

    print('Loading Dataset ...')
    imageSet_test = LystoTestset(filepath="data/testing.h5", transform=trans, tile_size=args.tile_size,
                                 interval=args.interval, num_of_imgs=20)

    test_tile(imageSet_test, batch_size=args.batch_size, workers=args.workers, output_path=args.output)
