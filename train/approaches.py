from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import *

from save_images import generate_masks


def train_tile(loader, epoch, total_epochs, model, device, criterion, optimizer, scheduler):
    """Tile training for one epoch.

    :param loader:          训练集的迭代器
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param criterion:       损失函数（criterion_cls）
    :param optimizer:       优化器
    """

    # tile training, dataset.mode = 3
    model.train()

    tile_num = 0
    train_loss = 0.
    train_bar = tqdm(loader, desc="tile training")
    for i, (data, label) in enumerate(train_bar):
        train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))

        output = model(data.to(device))
        optimizer.zero_grad()
        loss = criterion(output, label.to(device))  # CrossEntropy 本身携带了 softmax()

        tile_num += data.size(0)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    train_loss /= tile_num
    return train_loss


def train_image(loader, epoch, total_epochs, model, device, crit_cls, crit_reg, optimizer, scheduler, beta, gamma):
    """tile + image training for one epoch. image mode = image_cls + image_reg + image_seg

    :param loader:          训练集的迭代器
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param crit_cls:        分类器损失函数
    :param crit_reg:        回归损失函数
    :param optimizer:       优化器
    :param scheduler:       学习率调度器
    :param beta:            image_cls_loss 系数
    :param gamma:           image_reg_loss 系数
    """

    # image training, dataset.mode = 5
    model.train()

    image_cls_loss = 0.
    image_reg_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, desc="image training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (data, label_cls, label_num) in enumerate(train_bar):

        output = model(data.to(device))

        optimizer.zero_grad()

        image_cls_loss_i = crit_cls(output[0], label_cls.to(device))
        image_reg_loss_i = crit_reg(output[1].squeeze(), label_num.to(device, dtype=torch.float32))
        # # For weighted MSE
        # image_reg_loss_i = crit_reg(output[1].squeeze(), label_num.to(device, dtype=torch.float32),
        #                             label_cls.to(device, dtype=torch.float32))

        # total_loss_i = alpha * tile_loss_i + beta * image_cls_loss_i + \
        #                gamma * image_reg_loss_i + delta * image_seg_loss_i
        image_loss_i = beta * image_cls_loss_i + gamma * image_reg_loss_i
        image_loss_i.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_cls_loss += image_cls_loss_i.item() * data.size(0)
        image_reg_loss += image_reg_loss_i.item() * data.size(0)
        image_loss += image_loss_i.item() * data.size(0)

        # print("image data size:", data.size(0))

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()

    # print("Total images:", len(loader.dataset))

    image_loss /= len(loader.dataset)
    image_cls_loss /= len(loader.dataset)
    image_reg_loss /= len(loader.dataset)

    return image_cls_loss, image_reg_loss, image_loss


def train_alternative(loader, epoch, total_epochs, model, device, crit_cls, crit_reg, crit_seg, optimizer,
                      scheduler, threshold, alpha, beta, gamma, delta):
    """tile + image training for one epoch. image mode = image_cls + image_reg + image_seg

    :param loader:          训练集的迭代器
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param crit_cls:        分类器损失函数
    :param crit_reg:        回归损失函数
    :param crit_seg:        分割损失函数
    :param optimizer:       优化器
    :param scheduler:       学习率调度器
    :param alpha:           tile_loss 系数
    :param beta:            image_cls_loss 系数
    :param gamma:           image_reg_loss 系数
    :param delta:           image_seg_loss 系数
    """

    # alternative training, dataset.mode = 2

    tile_num = 0
    tile_loss = 0.
    image_cls_loss = 0.
    image_reg_loss = 0.
    image_seg_loss = 0.
    image_loss = 0.

    train_bar = tqdm(loader, desc="alternative training")
    train_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
    for i, (data, labels) in enumerate(train_bar):

        # pt.1: tile training
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

        # pt.2: image training
        model.setmode("image")
        model.train()
        output = model(data[0].to(device))

        optimizer.zero_grad()

        image_cls_loss_i = crit_cls(output[0], labels[0].to(device))
        image_reg_loss_i = crit_reg(output[1].squeeze(), labels[1].to(device, dtype=torch.float32))
        # # For weighted MSE
        # image_reg_loss_i = crit_reg(output[1].squeeze(), labels[1].to(device, dtype=torch.float32),
        #                             labels[0].to(device, dtype=torch.float32))
        # image_seg_loss_i = crit_seg(output[?], labels[?].to(device))

        # total_loss_i = alpha * tile_loss_i + beta * image_cls_loss_i + \
        #                gamma * image_reg_loss_i + delta * image_seg_loss_i
        image_loss_i = beta * image_cls_loss_i + gamma * image_reg_loss_i
        image_loss_i.backward()
        optimizer.step()
        if isinstance(scheduler, (CyclicLR, OneCycleLR)):
            scheduler.step()

        image_cls_loss += image_cls_loss_i.item() * data[0].size(0)
        image_reg_loss += image_reg_loss_i.item() * data[0].size(0)
        # image_seg_loss += image_seg_loss_i.item() * image_data[0].size(0)
        image_loss += image_loss_i.item() * data[0].size(0)

        # print("image data size:", data[0].size(0))
        # print("tile data size:", data[1].size(0))

        # if (i + 1) % ((len(loader) + 1) // mini_epochs?) == 0:
        #     train_seg(train_loader_forward, batch_size, epoch, total_epochs, model, device,
        #     crit_seg, optimizer, threshold, save_masks=False)
        #     # TODO: save masks of all data after n iterations

    if not (scheduler is None or isinstance(scheduler, (CyclicLR, OneCycleLR))):
        scheduler.step()
    # print("Total tiles:", tile_num)
    # print("Total images:", len(loader.dataset))

    tile_loss /= tile_num
    image_loss /= len(loader.dataset)
    image_cls_loss /= len(loader.dataset)
    image_reg_loss /= len(loader.dataset)
    # image_seg_loss /= len(loader.dataset)
    image_seg_loss = 0.
    return tile_loss, image_cls_loss, image_reg_loss, image_seg_loss, image_loss


# def train_seg(trainset, loader, batch_size, epoch, total_epochs, model, device, crit_seg, optimizer, threshold, save_masks):
#
#     # pt.3 segment training
#     trainset.setmode(1)
#     model.setmode("segment")
#
#     probs = torch.Tensor(len(loader.dataset))
#     with torch.no_grad():
#         tile_bar = tqdm(loader, desc="tile forwarding (seg)", leave=False)
#         tile_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
#         for i, input in enumerate(tile_bar):
#             model.setmode("tile")
#             model.eval()
#             output = model(input[0].to(device))
#             output = F.softmax(output, dim=1)
#             probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()
#
#     groups = np.array(trainset.tileIDX)
#     tiles = np.array(trainset.tiles_grid)
#
#     order = np.lexsort((probs, groups))
#     groups = groups[order]
#     probs = probs[order]
#     tiles = tiles[order]
#
#     index = [prob > threshold for prob in probs]
#
#     pseudo_masks = generate_masks(trainset, tiles[index], groups[index], save_masks)
#
#     # TODO: create a new dataset for masks
#     trainset.setmode(2)
#     model.train()
#     # TODO: load data for segmentation head training