import warnings
import os
import numpy as np
import argparse
import time
import cv2
import csv
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import model.resnet as models
from utils.collate import default_collate

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(prog="train.py", description='Training')
parser.add_argument('-e', '--epochs', type=int, default=10, help='total number of epochs to train (default: 10)')
parser.add_argument('--tile_only', action='store_true', help='if whole image mode is disabled')
parser.add_argument('-E', '--encoder', type=str, default='resnet18',
                    help='structure of the shared encoder, [resnet18 (default), resnet34, resnet50]')
parser.add_argument('-b', '--tile_batch_size', type=int, default=64, help='mini-batch size of tiles (default: 64)')
parser.add_argument('-B', '--image_batch_size', type=int, default=64, help='mini-batch size of images (default: 64)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-m', '--mini_epoch_size', default=1000, type=int, help='number of iterations per mini-epoch (default: 1000)')
parser.add_argument('--test_every', default=1, type=int, help='validate every (default: 1) epoch(s)')
parser.add_argument('-k', '--tiles_per_pos', default=1, type=int,
                    help='k tiles are from a single positive cell (default: 1, standard MIL)')
parser.add_argument('-n', '--topk_neg', default=30, type=int,
                    help='top k tiles from a negative image (default: 30, standard MIL)')
parser.add_argument('-t', '--tile_size', type=int, default=32, help='size of a certain tile (default: 32)')
parser.add_argument('--distributed', action="store_true",
                    help='if distributed parallel training is enabled (seems to no avail)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint', metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint)')
parser.add_argument('-r', '--resume', type=str, default=None, metavar='MODEL/FILE/PATH',
                    help='continue training from a checkpoint.pth')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
else:
    torch.manual_seed(1)

max_acc = 0
verbose = True
now = int(time.time())

trainset = None
valset = None

model = models.encoders[args.encoder]
# 把 ResNet 源码中的分为 1000 类改为二分类（由于预训练模型文件的限制，只能在外面改）
model.fc_tile = nn.Linear(model.fc_tile.in_features, 2)

last_epoch = 0
if args.resume:
    last_epoch = torch.load(args.resume)['epoch']
    model.load_state_dict(torch.load(args.resume)['state_dict'])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.ToTensor(), normalize])
# trans = transforms.ToTensor()

crit_cls = nn.CrossEntropyLoss()
crit_reg = nn.MSELoss()
crit_seg = None # TODO
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

if dist.is_nccl_available() and args.distributed:
    print('\nNCCL is available. Setup distributed parallel training with {} devices...\n'
          .format(torch.cuda.device_count()))
    dist.init_process_group(backend='nccl', world_size=1)
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device)
    model.to(device)

def train(batch_size, tile_only, workers, total_epochs, last_epoch, mini_epoch_size, test_every, model,
          crit_cls, crit_reg, crit_seg, optimizer, tiles_per_pos, topk_neg, output_path):
    """one training epoch = tile mode -> image mode

    :param batch_size:      DataLoader 打包的小 batch 大小
    :param workers:         DataLoader 使用的进程数
    :param total_epochs:    迭代总次数
    :param last_epoch:      上一次迭代的次数（当继续训练时）
    :param mini_epoch_size: 一个 mini_epoch 包含迭代的次数
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param crit_cls:        分类器损失函数
    :param crit_reg:        回归损失函数
    :param crit_seg:        分割损失函数
    :param optimizer:       优化器
    :param tiles_per_pos:   在**单个阳性细胞**上选取的图像块数 (topk_pos = tiles_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k tile **总数**
    :param output_path:     保存模型文件和训练数据结果的目录
    """

    global device, now

    # shuffle 只能是 False
    # 暂定对 tile 的训练和对 image 的训练所用的 batch_size 是一样的
    collate_fn = default_collate
    # TODO: how can I split the training step for distributed parallel training?
    train_sampler = DistributedSampler(trainset) if dist.is_nccl_available() and args.distributed else None
    val_sampler = DistributedSampler(valset) if dist.is_nccl_available() and args.distributed else None
    train_loader_forward = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                      sampler=train_sampler, pin_memory=True)
    train_loader_backward = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       sampler=train_sampler, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, sampler=val_sampler,
                            pin_memory=True)

    # open output file
    fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'w')
    fconv.write('epoch,tile_loss,image_cls_loss,image_reg_loss,image_seg_loss,total_loss\n')
    fconv.close()
    # 训练结果保存在 output_path/<timestamp>-training.csv
    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'w')
    fconv.write('epoch,tile_error,tile_fpr,tile_fnr,image_error,image_fpr,image_fnr,mae,mse\n')
    fconv.close()
    # 验证结果保存在 output_path/<timestamp>-validation.csv

    print('Start training ...')
    # if resume:
    #     print('Resuming from the checkpoint (epochs: {}).'.format(model['epoch']))

    start = int(time.time())
    with SummaryWriter() as writer:
        for epoch in range(1 + last_epoch, total_epochs + 1):

            # Forwarding step
            trainset.setmode(1)
            probs = inference_tiles(train_loader_forward, batch_size, epoch, total_epochs)
            sample(probs, tiles_per_pos, topk_neg)

            # Training tile-mode only
            if tile_only:
                trainset.setmode(3)
                loss = train_tile(train_loader_forward, batch_size, epoch, total_epochs, model, crit_cls, optimizer)
                print("tile loss: {:.4f}".format(loss))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                writer.add_scalar("tile loss", loss, epoch)

                # Validating step
                if (epoch + 1) % test_every == 0:
                    valset.setmode(1)
                    print('Validating ...')

                    probs_p = inference_tiles(val_loader, batch_size, epoch, total_epochs)
                    metrics_p = validation_tile(probs_p)
                    print('tile error: {} | tile FPR: {} | tile FNR: {}'.format(*metrics_p))

                    writer.add_scalar('tile error rate', metrics_p[0], epoch)
                    writer.add_scalar('tile false positive rate', metrics_p[1], epoch)
                    writer.add_scalar('tile false negative rate', metrics_p[2], epoch)
                    
                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{}\n'.format(epoch, *metrics_p))
                    fconv.close()

                    # 每验证一次，保存模型
                    obj = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'encoder': model.encoder_name,
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(output_path, 'checkpoint_{}epochs_tileonly.pth'.format(epoch)))

            # Alternative training step
            else:
                trainset.setmode(2)
                # if epoch == total_epochs:
                #     trainset.visualize_bboxes()  # tile visualize testing
                alpha = 1.
                beta = 0.1
                gamma = 0.9
                delta = 0.1
                loss = train_alternative(train_loader_backward, batch_size, epoch, total_epochs, mini_epoch_size, model, crit_cls,
                                         crit_reg, crit_seg, optimizer, beta, gamma, delta)

                print("tile loss: {:.4f} | image cls loss: {:.4f} | image reg loss: {:.4f} | image seg loss: {:.4f} | image loss: {:.4f}"
                      .format(*loss))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},{},{},{},{},{}\n'.format(epoch, *loss))
                fconv.close()

                writer.add_scalar("tile loss", loss[0], epoch)
                writer.add_scalar("image cls loss", loss[1], epoch)
                writer.add_scalar("image reg loss", loss[2], epoch)
                writer.add_scalar("image seg loss", loss[3], epoch)
                writer.add_scalar("image loss", loss[4], epoch)

                # Validating step
                if (epoch + 1) % test_every == 0:
                    valset.setmode(1)
                    print('Validating ...')

                    probs_p = inference_tiles(val_loader, batch_size, epoch, total_epochs)
                    metrics_p = validation_tile(probs_p)
                    # TODO: heatmap validation
                    print('tile error: {} | tile FPR: {} | tile FNR: {}'.format(*metrics_p))

                    writer.add_scalar('tile error rate', metrics_p[0], epoch)
                    writer.add_scalar('tile false positive rate', metrics_p[1], epoch)
                    writer.add_scalar('tile false negative rate', metrics_p[2], epoch)

                    # image validating
                    valset.setmode(4)
                    probs_s, reg, seg = predict_image(val_loader, batch_size, epoch, total_epochs)
                    metrics_s = validation_image(probs_s, reg, seg)
                    print('image error: {} | image FPR: {} | image FNR: {}\nMAE: {} | MSE: {}\n'.format(*metrics_s))
                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{},{},{},{},{},{}\n'.format(epoch, *(metrics_p + metrics_s)))
                    fconv.close()

                    writer.add_scalar('image error rate', metrics_s[0], epoch)
                    writer.add_scalar('image false positive rate', metrics_s[1], epoch)
                    writer.add_scalar('image false negative rate', metrics_s[2], epoch)
                    writer.add_scalar('image mae', metrics_s[3], epoch)
                    writer.add_scalar('image mse', metrics_s[4], epoch)

                    # 每验证一次，保存模型
                    obj = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'encoder': model.encoder_name,
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(output_path, 'checkpoint_{}epochs.pth'.format(epoch)))

    end = int(time.time())
    print("\nTrained for {} epochs. Runtime: {}s".format(total_epochs, end - start))

def inference_tiles(loader, batch_size, epoch, total_epochs):
    """前馈推导一次模型，获取实例分类概率。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    """
    global device

    model.setmode("tile")
    model.eval()

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        tile_bar = tqdm(loader, total=len(loader) + 1)
        for i, input in enumerate(tile_bar):
            tile_bar.set_postfix(step="tile forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader) + 1))
            # softmax 输出 [[a,b],[c,d]] shape = batch_size*2
            output = model(input[0].to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] 取出 softmax 得到的概率，产生：[b, d, ...]
            # input.size(0) 返回 batch 中的实例数量
            probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def sample(probs, tiles_per_pos, topk_neg):
    """找出概率为 top-k 的图像块，制作迭代使用的数据集。

    :param probs:           inference_tiles() 得到的各个 tile 概率
    :param tiles_per_pos:   在**单个阳性细胞**上选取的图像块数 (topk_pos = tiles_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k tile **总数**
    """

    global verbose

    groups = np.array(trainset.tileIDX)
    order = np.lexsort((probs, groups))

    index = np.empty(len(trainset), 'bool')
    for i in range(len(trainset)):
        topk = topk_neg if trainset.labels[groups[i]] == 0 else trainset.labels[groups[i]] * tiles_per_pos
        index[i] = groups[i] != groups[(i + topk) % len(groups)]

    p, n = trainset.make_train_data(list(order[index]))
    if verbose:
        print("Training data is sampled. (Pos samples: {} | Neg samples: {})".format(p, n))


def predict_image(loader, batch_size, epoch, total_epochs):
    """前馈推导一次模型，获取图像级的分类概率和回归预测值。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :return:                切片分类概率，细胞计数，分割结果
    """

    model.setmode("image")
    model.eval()

    probs = torch.tensor(())
    nums = torch.tensor(())
    feats = torch.tensor(())
    with torch.no_grad():
        image_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, (data, label_cls, label_num, _) in enumerate(image_bar):
            image_bar.set_postfix(step="image forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            output = model(data.to(device))
            output_cls = F.softmax(output[0], dim=1)
            probs = torch.cat((probs, output_cls.detach()[:, 1].clone().cpu()), dim=0)
            nums = torch.cat((nums, output[1].detach()[:, 0].clone().cpu()), dim=0)
            feats = torch.cat((feats, output[2].detach().clone().cpu()), dim=0)
    return probs.numpy(), nums.numpy(), feats.numpy()


def train_tile(loader, batch_size, epoch, total_epochs, model, criterion, optimizer):
    """Tile training for one epoch.

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param criterion:       用于补丁级训练的损失函数（criterion_cls）
    :param optimizer:       优化器
    """
    global device

    model.train()

    tile_num = 0
    train_loss = 0.
    train_bar = tqdm(loader, total=len(loader))
    for i, (data, label) in enumerate(train_bar):
        train_bar.set_postfix(step="tile training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))

        output = model(data.to(device))
        optimizer.zero_grad()
        loss = criterion(output, label.to(device))  # CrossEntropy 本身携带了 softmax()

        tile_num += data.size(0)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    train_loss /= tile_num
    return train_loss


def train_alternative(loader, batch_size, epoch, total_epochs, mini_epoch_size, model, crit_cls, crit_reg, crit_seg, optimizer,
                      alpha, beta, gamma, delta):
    """tile + image training for one epoch. image mode = image_cls + image_reg + image_seg

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param mini_epoch_size: 一个 mini_epoch 包含迭代的次数
    :param model:           网络模型
    :param crit_cls:        分类器损失函数
    :param crit_reg:        回归损失函数
    :param crit_seg:        分割损失函数
    :param optimizer:       优化器
    :param alpha:           tile_loss 系数
    :param beta:            image_cls_loss 系数
    :param gamma:           image_reg_loss 系数
    :param delta:           image_seg_loss 系数
    """

    global device

    tile_num = 0
    tile_loss = 0.
    image_cls_loss = 0.
    image_reg_loss = 0.
    image_seg_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, total=len(loader) + 1)
    for i, (data, labels) in enumerate(train_bar):
        train_bar.set_postfix(step="alternative training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader) + 1))

        # tile training
        model.setmode("tile")
        model.train()
        # print("images pack size:", data[0].size())
        # print("tiles pack size:", data[1].size())
        output = model(data[1].to(device))
        optimizer.zero_grad()

        tile_loss_i = alpha * crit_cls(output, labels[2].to(device))
        tile_loss_i.backward()
        optimizer.step()

        tile_loss += tile_loss_i.item() * data[1].size(0)
        tile_num += data[1].size(0)

        # model.eval()
        # with torch.no_grad():
        #     output = model(data[1].to(device)) # ?
        #     output = F.softmax(output, dim=1)
        #     probs = output.detach()[:, 1].clone().cpu().numpy() # 当前 batch 中的图像前馈得到的 tile 概率

        # image training
        model.setmode("image")
        model.train()
        output = model(data[0].to(device))

        optimizer.zero_grad()

        image_cls_loss_i = crit_cls(output[0], labels[0].to(device))
        image_reg_loss_i = crit_reg(output[1].squeeze(), labels[1].to(device, dtype=torch.float32))
        # image_seg_loss_i = crit_seg(output[?], labels[?].to(device))

        # total_loss_i = alpha * tile_loss_i + beta * image_cls_loss_i + \
        #                gamma * image_reg_loss_i + delta * image_seg_loss_i
        image_loss_i = beta * image_cls_loss_i + gamma * image_reg_loss_i
        image_loss_i.backward()
        optimizer.step()

        image_cls_loss += image_cls_loss_i.item() * data[0].size(0)
        image_reg_loss += image_reg_loss_i.item() * data[0].size(0)
        # image_seg_loss += image_seg_loss_i.item() * image_data[0].size(0)
        image_loss += image_loss_i.item() * data[0].size(0)

        # print("image data size:", data[0].size(0))
        # print("tile data size:", data[1].size(0))

        if (i + 1) % mini_epoch_size == 0:
            # TODO: save masks after n iterations
            pass

        # TODO: read masks for seg head training



    # print("Total tiles:", tile_num)
    # print("Total images:", len(loader.dataset))

    tile_loss /= tile_num
    image_loss /= len(loader.dataset)
    image_cls_loss /= len(loader.dataset)
    image_reg_loss /= len(loader.dataset)
    # image_seg_loss /= len(loader.dataset)
    image_seg_loss = 0.
    return tile_loss, image_cls_loss, image_reg_loss, image_seg_loss, image_loss

# def validation_tile(probs):
#     """tile mode 的验证"""
#
#     val_groups = np.array(valset.tileIDX)
#
#     max_prob = np.empty(len(valset.labels))  # 模型预测的实例最大概率列表，每张切片取最大概率的 tile
#     max_prob[:] = np.nan
#     order = np.lexsort((probs, val_groups))
#     # 排序
#     val_groups = val_groups[order]
#     val_probs = probs[order]
#     # 取最大
#     val_index = np.empty(len(val_groups), 'bool')
#     val_index[-1] = True
#     val_index[:-1] = val_groups[1:] != val_groups[:-1]
#     max_prob[val_groups[val_index]] = val_probs[val_index]
#
#     # 计算错误率、FPR、FNR
#     probs = np.round(max_prob)  # 每张切片由最大概率的 tile 得到的标签
#     err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
#     return err, fpr, fnr

def validation_tile(probs):
    """tile mode 的验证"""

    thr = 0.88  # tile 的分类阈值，根据经验设定
    val_groups = np.array(valset.tileIDX)

    order = np.lexsort((probs, val_groups)) # 对 tile 按预测概率排序
    val_groups = val_groups[order]
    val_probs = probs[order]

    val_index = np.array([prob > thr for prob in val_probs])

    # 制作分类用的 label：根据计数标签 = n，前 n 个 tile 为阳性
    labels = np.zeros(len(val_probs))
    for i in range(1, len(val_probs) + 1):
        if i == len(val_probs) or val_groups[i] != val_groups[i - 1]:
            labels[i - valset.labels[val_groups[i - 1]]: i] = [1] * valset.labels[val_groups[i - 1]]

    # 计算错误率、FPR、FNR
    err, fpr, fnr = calc_err(val_index, labels)
    return err, fpr, fnr

def heatmap(valset, tiles, probs, groups, output_path):
    """把预测得到的阳性细胞区域标在图上。

    :param valset:          验证集
    :param tiles:         要标注的补丁
    :param probs:           补丁对应的概率
    :param output_path:     图像存储路径
    """

    count = 0
    # test_idx = len(valset)
    test_idx = 20

    for i in range(1, len(groups) + 1):
        count += 1

        if i == len(groups) or groups[i] != groups[i - 1]:
            img = valset.images[groups[i - 1]]
            mask = np.zeros((img.shape[0], img.shape[1]))

            for j in range(i - count, i):
                tile_mask = np.full((valset.size, valset.size), probs[j])
                grid = list(map(int, tiles[j]))

                mask[grid[0]: grid[0] + valset.size,
                     grid[1]: grid[1] + valset.size] = tile_mask

                # 输出信息
                # print("prob_{}:{}".format(groups[i - 1], probs[j]))
                fconv = open(os.path.join(output_path, '{}-pred.csv'.format(now)), 'a', newline="")
                w = csv.writer(fconv)
                w.writerow([groups[i - 1], '{}'.format(grid), probs[j]])
                fconv.close()

            mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "validate_{}.png".format(groups[i - 1])))

            count = 0

            # 没有阳性 tile 的时候。。。
            if i == len(groups) and groups[i - 1] != test_idx or groups[i - 1] != groups[i] - 1:
                for j in range(groups[i - 1] + 1, test_idx if i == len(groups) else groups[i]):
                    img = valset.images[j]
                    mask = np.zeros((img.shape[0], img.shape[1]))
                    mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
                    img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
                    Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "validate_{}.png".format(j)))

def validation_image(probs, reg, seg):
    """image mode 的验证"""

    # probs = np.round(probs)  # go soft?
    # TODO: is it necessary to validate image classification?
    # err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
    err = fpr = fnr = 0
    mae = metrics.mean_absolute_error(valset.labels, reg)
    mse = metrics.mean_squared_error(valset.labels, reg)
    return err, fpr, fnr, mae, mse

def calc_err(pred, real):
    """计算分类任务的错误率、假阳性率、假阴性率"""

    pred = np.asarray(pred)
    real = np.asarray(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0] # 错误率 = 预测错误的和 / 总和
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum() # 假阳性率 = 假阳性 / 所有的阴性
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum() # 假阴性率 = 假阴性 / 所有的阳性
    return err, fpr, fnr


if __name__ == "__main__":
    from dataset.dataset import LystoDataset

    print("Training settings: ")
    print("Training Mode: {} | Device: {} | Encoder: {} | {} epochs in total | Validate every {} epoch(s) | Image batch size: {} | Negative top-k: {}"
          .format("tile" if args.tile_only else "tile + image", 'GPU' if torch.cuda.is_available() else 'CPU',
                  args.encoder, args.epochs, args.test_every, args.image_batch_size, args.topk_neg))

    print('Loading Dataset ...')
    trainset = LystoDataset(filepath="data/training.h5", transform=trans)
    valset = LystoDataset(filepath="data/training.h5", train=False, transform=trans)

    train(batch_size=args.image_batch_size,
          tile_only=args.tile_only,
          workers=args.workers,
          total_epochs=args.epochs,
          last_epoch=last_epoch,
          mini_epoch_size=args.mini_epoch_size,
          test_every=args.test_every,
          model=model,
          crit_cls=crit_cls,
          crit_reg=crit_reg,
          crit_seg=crit_seg,
          optimizer=optimizer,
          tiles_per_pos=args.tiles_per_pos,
          topk_neg=args.topk_neg,
          output_path=args.output
          )
