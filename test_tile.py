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
import torch.nn.functional as F
import torchvision.transforms as transforms

import model.resnet as models

now = int(time.time())

parser = argparse.ArgumentParser(prog="test_tile.py", description='Testing & Heatmap')
parser.add_argument('-m', '--model', type=str, default='checkpoint/checkpoint_10epochs.pth',
                    help='path to pretrained model (default: checkpoint/checkpoint_10epochs.pth)')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='mini-batch size (default: 64)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
# parser.add_argument('-k', '--topk', default=30, type=int,
#                     help='top k tiles are assumed to be of the same class as the image (default: 10, standard MIL)')
parser.add_argument('-i', '--interval', type=int, default=5, help='sample interval of tiles (default: 5)')
parser.add_argument('-p', '--tile_size', type=int, default=32, help='size of each tile (default: 32)')
parser.add_argument('-c', '--threshold', type=float, default=0.88,
                    help='minimal prob for tiles to show in heatmap (default: 0.88)')
parser.add_argument('-d', '--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('-o', '--output', type=str, default='./output/{}/'.format(now),
                    help='path of output details .csv file')

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

model = models.encoders[torch.load(args.model)['encoder']]
model.fc_tile = nn.Linear(model.fc_tile.in_features, 2)
epoch = torch.load(args.model)['epoch']
model.load_state_dict(torch.load(args.model)['state_dict'])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.ToTensor(), normalize])
# trans = transforms.ToTensor()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
model.to(device)


def test(testset, batch_size, workers, model, output_path):
    """
    :param testset:         测试数据集
    :param batch_size:      Dataloader 打包的小 batch 大小
    :param workers:         Dataloader 使用的进程数
    :param model:           网络模型
    :param output_path:     保存模型文件的目录
    """

    global epoch

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    # 热图中各个 tile 的信息保存在 output_path/<timestamp>-pred-<epoch>-<tilesize>-<interval>-<threshold>.csv
    fconv = open(os.path.join(output_path, '{}-pred-e{}-p{}-i{}-c{}.csv'.format(
        now, epoch, args.tile_size, args.interval, args.threshold)), 'w', newline="")
    w = csv.writer(fconv)
    w.writerow(['tile_size', '{}'.format(testset.tile_size)])
    w.writerow(['interval', '{}'.format(testset.interval)])
    w.writerow(['idx', 'grid', 'prob'])
    fconv.close()

    print('Start testing ...')

    model.setmode("tile")
    model.eval()
    probs = predict_tiles(test_loader, batch_size)
    tiles, probs, groups = rank(testset, probs)

    # 生成热图
    heatmap(testset, tiles, probs, groups, output_path)


def predict_tiles(loader, batch_size):
    """前馈推导一次模型，获取实例分类概率。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    """
    global device

    model.setmode("tile")
    model.eval()

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        tile_bar = tqdm(loader, total=len(loader) + 1)
        for i, input in enumerate(tile_bar):
            tile_bar.set_postfix(step="tile forwarding",
                                  batch="[{}/{}]".format(i + 1, len(loader) + 1))
            # softmax 输出 [[a,b],[c,d]] shape = batch_size*2
            output = model(input.to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] 取出 softmax 得到的概率，产生：[b, d, ...]
            # input.size(0) 返回 batch 中的实例数量
            probs[i * batch_size:i * batch_size + input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def rank(testset, probs):
    """寻找最大概率的 tile ，用于作图。

    :param testset:     测试集
    :param probs:       求得的概率
    :return:            取出的补丁以及对应的概率
    """

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

# def heatmap(testset, tiles, probs, topk, output_path):
#     """把预测得到的阳性细胞区域标在图上。
#
#     :param testset:         测试集
#     :param tiles:           要标注的补丁
#     :param probs:           补丁对应的概率
#     :param topk:            标注的补丁数
#     :param output_path:     图像存储路径
#     """
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
#             fconv = open(os.path.join(output_path, 'pred.csv'), 'a', newline="")
#             w = csv.writer(fconv)
#             w.writerow([i, '{}'.format(grid), probs[idx + i * topk]])
#             fconv.close()
#
#         mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
#         img = img * 0.5 + mask * 0.5
#         Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(i)))


def heatmap(testset, tiles, probs, groups, output_path):
    """把预测得到的阳性细胞区域标在图上。

    :param testset:         测试集
    :param tiles:           要标注的补丁
    :param probs:           补丁对应的概率
    :param output_path:     图像存储路径
    """

    global epoch

    count = 0
    # test_idx = len(testset)
    test_idx = 20

    for i in range(1, len(groups) + 1):
        count += 1

        if i == len(groups) or groups[i] != groups[i - 1]:
            img = testset.images[groups[i - 1]]
            mask = np.zeros(testset.image_size)

            for j in range(i - count, i):
                tile_mask = np.full((testset.tile_size, testset.tile_size), probs[j])
                grid = list(map(int, tiles[j]))

                mask[grid[0]: grid[0] + testset.tile_size,
                     grid[1]: grid[1] + testset.tile_size] = tile_mask

                # 输出信息
                # print("prob_{}:{}".format(groups[i - 1], probs[j]))
                fconv = open(os.path.join(output_path, '{}-pred-e{}-p{}-i{}-c{}.csv'.format(
                    now, epoch, args.tile_size, args.interval, args.threshold)), 'a', newline="")
                w = csv.writer(fconv)
                w.writerow([groups[i - 1], '{}'.format(grid), probs[j]])
                fconv.close()

            mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(groups[i - 1])))

            count = 0

            # 没有阳性 tile 的时候。。。
            if i == len(groups) and groups[i - 1] != test_idx or groups[i - 1] != groups[i] - 1:
                for j in range(groups[i - 1] + 1, test_idx if i == len(groups) else groups[i]):
                    img = testset.images[j]
                    mask = np.zeros((img.shape[0], img.shape[1]))
                    mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
                    img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
                    Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(j)))


if __name__ == "__main__":
    from dataset.dataset import LystoTestset

    print('Loading Dataset ...')
    imageSet_test = LystoTestset(filepath="data/testing.h5", transform=trans,
                                 interval=args.interval, size=args.tile_size, num_of_imgs=20)

    test(imageSet_test, batch_size=args.batch_size, workers=args.workers, model=model, output_path=args.output)
