import warnings
import os
import sys
import configparser
import argparse
import time
import csv
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import LystoDataset
from model import nets
from inference import inference_image
from train import train_image, WeightedMSELoss
from evaluate import evaluate_image
from utils import default_collate

warnings.filterwarnings("ignore")
now = int(time.time())

# Training settings
parser = argparse.ArgumentParser(prog="train_image.py", description='pt.1: image assessment training.')
parser.add_argument('-e', '--epochs', type=int, default=50,
                    help='total number of epochs to train (default: 50)')
parser.add_argument('--reg_only', action="store_true", help='only enable image regression head')
parser.add_argument('-H', '--hard_threshold', type=int, default=None,
                    help='Dynamically increase ratio of hard data by resampling between training epochs. '
                         'Hard data threshold defined by categorizing error / counting error (--reg_only). '
                         '(no dynamic training as default)')
parser.add_argument('-O', '--organ', type=str, default=None,
                    help='specify the category of training data {\'colon\', \'breast\', \'prostate\'} '
                         '(train all data as default)')
parser.add_argument('-E', '--encoder', type=str, default='resnet50',
                    help='structure of the shared encoder, {\'resnet18\', \'resnet34\', \'resnet50\' (default), '
                         '\'efficientnet_b0\', \'efficientnet_b2\', \'resnext50\', \'resnext101\'}')
parser.add_argument('-B', '--image_batch_size', type=int, default=48,
                    help='batch size of images (default: 48, 32 recommended for EfficientNet)')
parser.add_argument('-l', '--lr', type=float, default=8e-5, metavar='LR',
                    help='learning rate (8e-5 recommended for EfficientNet)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='decay of weight (default: 1e-4)')
parser.add_argument('-s', '--scheduler', type=str, default=None,
                    help='learning rate scheduler if necessary, '
                         '{\'OneCycleLR\', \'ExponentialLR\', \'CosineAnnealingWarmRestarts\'} (default: None)')
parser.add_argument('-a', '--augment', action="store_true", help='apply data augmentation')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int,
                    help='validate every (default: 1) epoch(s). To use all data for training, '
                         'set this greater than --epochs')
parser.add_argument('--distributed', action="store_true",
                    help='if distributed parallel training is enabled (seems to be no avail)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint/{}'.format(now), metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint/<timestamp>)')
parser.add_argument('-r', '--resume', type=str, default=None, metavar='MODEL/FILE/PATH',
                    help='continue training from a checkpoint.pth')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()


def train_cls(total_epochs, last_epoch, test_every, model, device, crit_cls, optimizer, scheduler,
              output_path):
    from train import train_image_cls
    from inference import inference_image_cls

    # open output file
    fconv = open(os.path.join(output_path, '{}-image-training.csv'.format(now)), 'w')
    fconv.write('epoch,image_cls_loss\n')
    fconv.close()
    # 训练结果保存在 output_path/<timestamp>-image-training.csv

    validate = lambda epoch, test_every: (epoch + 1) % test_every == 0
    start = int(time.time())
    with SummaryWriter(comment=output_path.rsplit('/', maxsplit=1)[-1]) as writer:
        print("PT.I - image classifier training ...")
        for epoch in range(1 + last_epoch, total_epochs + 1):
            try:
                if device.type == 'cuda':
                    torch.cuda.manual_seed(epoch)
                else:
                    torch.manual_seed(epoch)

                trainset.setmode(5)

                loss = train_image_cls(train_loader, epoch, total_epochs, model, device, crit_cls, optimizer, scheduler)

                print("image cls loss: {:.4f}".format(loss))
                fconv = open(os.path.join(output_path, '{}-image-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                writer.add_scalar("image cls loss", loss, epoch)

                # Validating step
                if validate(epoch, test_every):
                    print('Validating ...')

                    # image validating
                    valset.setmode(4)
                    categories = inference_image_cls(val_loader, model, device, epoch, total_epochs)

                    regconv = open(os.path.join(output_path, '{}-category-e{}.csv'.format(
                        now, epoch)), 'w', newline="")
                    w = csv.writer(regconv, delimiter=',')
                    w.writerow(['id', 'organ', 'label', 'category', 'cat_label', 'loss'])
                    for i, c in enumerate(categories):
                        w.writerow([i + 1, valset.organs[i], valset.labels[i], c, valset.cls_labels[i],
                                    np.abs(c - valset.cls_labels[i])])
                    regconv.close()

                save_model(epoch, model, optimizer, scheduler, output_path, prefix='cls_pt1')

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path, prefix='cls_pt1')
                print("\nTraining interrupted at epoch {}. Model saved in \'{}\'.".format(epoch, output_path))
                sys.exit(0)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def train_reg(total_epochs, last_epoch, test_every, model, device, crit_reg, optimizer, scheduler,
              output_path, *, thresh=None):
    from train import train_image_reg
    from inference import inference_image_reg

    global train_loader

    # open output file
    fconv = open(os.path.join(output_path, '{}-image-training.csv'.format(now)), 'w')
    fconv.write('epoch,image_reg_loss\n')
    fconv.close()
    # 训练结果保存在 output_path/<timestamp>-image-training.csv
    if test_every <= args.epochs:
        fconv = open(os.path.join(output_path, '{}-image-validation.csv'.format(now)), 'w')
        fconv.write('epoch,mse,qwk\n')
        fconv.close()
    # 验证结果保存在 output_path/<timestamp>-image-validation.csv

    if thresh is not None:
        scoringset = LystoDataset(os.path.join(training_data_path, "training.h5"), train=False, organ=trainset.organ,
                                  kfold=None)
        scoring_loader = DataLoader(scoringset, batch_size=train_loader.batch_size, shuffle=False,
                                    num_workers=train_loader.num_workers, pin_memory=True)

    print('Training ...' if not args.resume else 'Resuming from the checkpoint (epoch {})...'.format(last_epoch))

    validate = lambda epoch, test_every: (epoch + 1) % test_every == 0
    start = int(time.time())
    with SummaryWriter(comment=output_path.rsplit('/', maxsplit=1)[-1]) as writer:

        print("PT.I - image regression training ...")
        for epoch in range(1 + last_epoch, total_epochs + 1):
            try:

                if device.type == 'cuda':
                    torch.cuda.manual_seed(epoch)
                else:
                    torch.manual_seed(epoch)

                trainset.setmode(5)

                loss = train_image_reg(train_loader, epoch, total_epochs, model, device, crit_reg, optimizer, scheduler)

                print("image reg loss: {:.4f}".format(loss))
                fconv = open(os.path.join(output_path, '{}-image-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                writer.add_scalar("image reg loss", loss, epoch)

                # Validating step
                if validate(epoch, test_every):
                    print('Validating ...')

                    # image validating
                    valset.setmode(4)
                    counts = inference_image_reg(val_loader, model, device, epoch, total_epochs)

                    regconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(now, epoch)),
                                   'w', newline="")
                    w = csv.writer(regconv, delimiter=',')
                    w.writerow(['id', 'organ', 'label', 'count', 'category label', 'loss'])
                    for i, count in enumerate(np.round(counts).astype(int)):
                        w.writerow([i + 1, valset.organs[i], valset.labels[i], count, valset.cls_labels[i],
                                    np.abs(count - valset.labels[i])])
                    regconv.close()

                    metrics_i = evaluate_image(valset, [], counts)
                    print('image categories mAP: {} | MSE: {} | QWK: {}\n'.format(*metrics_i))
                    fconv = open(os.path.join(output_path, '{}-image-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{}\n'.format(epoch, *metrics_i[1:]))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_i)

                if thresh is not None:
                    print('Reconstructing training data ...')
                    scoringset.setmode(4)
                    counts = inference_image_reg(scoring_loader, model, device, epoch, total_epochs)
                    hard_indices = []

                    for i in range(len(counts)):
                        if abs(counts[i] - scoringset.labels[i]) >= thresh:
                            hard_indices.append(i)

                    trainset.random_delete(len(hard_indices))
                    for i in range(len(hard_indices)):
                        trainset.add_data(scoringset.organs[i], scoringset.images[i], scoringset.labels[i])
                    train_loader = DataLoader(trainset, batch_size=train_loader.batch_size, shuffle=True,
                                              num_workers=train_loader.num_workers,
                                              pin_memory=True, collate_fn=collate_fn)

                    print('Done. {0} removed & {0} added as hard data.'.format(len(hard_indices)))
                    for hi in hard_indices:
                        print('id: {}\tpred: {}\tgt: {}\tclass: {}'.format(hi,
                                                                           np.round(counts[hi]).astype(int),
                                                                           scoringset.labels[hi],
                                                                           scoringset.cls_labels[hi]))

                save_model(epoch, model, optimizer, scheduler, output_path, prefix='reg_pt1')

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path, prefix='reg_pt1')
                print("\nTraining interrupted at epoch {}. Model saved in \'{}\'.".format(epoch, output_path))
                sys.exit(0)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def train(total_epochs, last_epoch, test_every, model, device, crit_cls, crit_reg, optimizer, scheduler,
          output_path, *, thresh=None):
    """pt.1: image assessment training.

    :param total_epochs:    迭代总次数
    :param last_epoch:      上一次迭代的次数（当继续训练时）
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param device:          模型所在的设备
    :param crit_cls:        分类损失函数
    :param crit_reg:        回归损失函数
    :param optimizer:       优化器
    :param scheduler:       学习率调度器
    :param output_path:     保存模型文件和训练数据结果的目录
    """

    global train_loader

    # open output file
    fconv = open(os.path.join(output_path, '{}-image-training.csv'.format(now)), 'w')
    fconv.write('epoch,image_cls_loss,image_reg_loss,image_loss,image_seg_loss\n')
    fconv.close()
    # 训练结果保存在 output_path/<timestamp>-image-training.csv
    if test_every <= args.epochs:
        fconv = open(os.path.join(output_path, '{}-image-validation.csv'.format(now)), 'w')
        fconv.write('epoch,image_map,mse,qwk\n')
        fconv.close()
    # 验证结果保存在 output_path/<timestamp>-image-validation.csv

    if thresh is not None:
        scoringset = LystoDataset(os.path.join(training_data_path, "training.h5"), train=False, organ=trainset.organ,
                                  kfold=None)
        scoring_loader = DataLoader(scoringset, batch_size=train_loader.batch_size, shuffle=False,
                                    num_workers=train_loader.num_workers, pin_memory=True)

    print('Training ...' if not args.resume else 'Resuming from the checkpoint (epoch {})...'.format(last_epoch))

    validate = lambda epoch, test_every: (epoch + 1) % test_every == 0
    start = int(time.time())
    with SummaryWriter(comment=output_path.rsplit('/', maxsplit=1)[-1]) as writer:
        alpha = 1
        beta = 1

        print("PT.I - image assessment training ...")
        for epoch in range(1 + last_epoch, total_epochs + 1):
            try:
                if device.type == 'cuda':
                    torch.cuda.manual_seed(epoch)
                else:
                    torch.manual_seed(epoch)

                trainset.setmode(5)

                loss = train_image(train_loader, epoch, total_epochs, model, device, crit_cls, crit_reg,
                                   optimizer, scheduler, alpha, beta)

                print("image cls loss: {:.4f} | image reg loss: {:.4f} | image loss: {:.4f}"
                      .format(*loss))
                fconv = open(os.path.join(output_path, '{}-image-training.csv'.format(now)), 'a')
                fconv.write('{},{},{},{}\n'.format(epoch, *loss))
                fconv.close()

                add_scalar_loss(writer, epoch, loss)

                # Validating step
                if validate(epoch, test_every):
                    print('Validating ...')

                    # image validating
                    valset.setmode(4)
                    categories, counts = inference_image(val_loader, model, device, epoch, total_epochs)

                    regconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
                        now, epoch)), 'w', newline="")
                    w = csv.writer(regconv, delimiter=',')
                    w.writerow(['id', 'organ', 'label', 'count', 'category', 'loss'])
                    for i, count in enumerate(counts):
                        w.writerow([i + 1, valset.organs[i], valset.labels[i], count, valset.cls_labels[i],
                                    np.abs(count - valset.labels[i])])
                    regconv.close()

                    metrics_i = evaluate_image(valset, categories, counts)
                    print('image categories mAP: {} | MSE: {} | QWK: {}\n'.format(*metrics_i))
                    fconv = open(os.path.join(output_path, '{}-image-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{}\n'.format(epoch, *metrics_i))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_i)

                if thresh is not None:
                    print('Reconstructing training data ...')
                    scoringset.setmode(4)
                    categories, counts = inference_image(scoring_loader, model, device, epoch, total_epochs)
                    hard_indices = []

                    # for i in range(len(categories)):
                    #    if abs(categories[i] - scoringset.cls_labels[i]) >= thresh:
                    #        hard_indices.append(i)
                    for i in range(len(counts)):
                        if abs(counts[i] - scoringset.labels[i]) >= thresh:
                            hard_indices.append(i)

                    trainset.random_delete(len(hard_indices))
                    for i in range(len(hard_indices)):
                        trainset.add_data(scoringset.organs[i], scoringset.images[i], scoringset.labels[i])
                    train_loader = DataLoader(trainset, batch_size=train_loader.batch_size, shuffle=True,
                                              num_workers=train_loader.num_workers,
                                              pin_memory=True, collate_fn=collate_fn)

                    print('Done. {0} removed & {0} added as hard data.'.format(len(hard_indices)))
                    for hi in hard_indices:
                        print('id: {}\tpred: {}/{}\tgt: {}/{}'.format(hi,
                                                                           np.round(categories[hi]).astype(int),
                                                                           np.round(counts[hi]).astype(int),
                                                                           scoringset.cls_labels[hi],
                                                                           scoringset.labels[hi]))

                save_model(epoch, model, optimizer, scheduler, output_path)

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path)
                print("\nTraining interrupted at epoch {}. Model saved in \'{}\'.".format(epoch, output_path))
                sys.exit(0)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def save_model(epoch, model, optimizer, scheduler, output_path, prefix='pt1'):
    """用 .pth 格式保存模型。"""
    # save params of resnet encoder and image head only
    state_dict = OrderedDict({k: v for k, v in model.state_dict().items()
                              if k.startswith(model.encoder_prefix +
                                              model.image_module_prefix)})
    obj = {
        'mode': 'image',
        'epoch': epoch,
        'state_dict': state_dict,
        'encoder': model.encoder_name,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(obj, os.path.join(output_path, '{}_{}epochs.pth'.format(prefix, epoch)))


def add_scalar_loss(writer, epoch, losses):
    writer.add_scalar("image cls loss", losses[0], epoch)
    writer.add_scalar("image reg loss", losses[1], epoch)
    writer.add_scalar("image loss", losses[2], epoch)


def add_scalar_metrics(writer, epoch, metrics):
    metrics = list(metrics)

    assert len(metrics) == 3, "Image metrics should include 3 items: mAP, MSE and QWK. "
    writer.add_scalar('image map', metrics[0], epoch)
    writer.add_scalar('image mse', metrics[1], epoch)
    writer.add_scalar('image qwk', metrics[2], epoch)


if __name__ == "__main__":
    print("Training settings: ")
    print("Training Mode: {} | Device: {} | {} | {} epoch(s) in total\n"
          "{} | Initial LR: {} | Output directory: {}"
          .format('tile + image (pt.1{})'.format(', DYNAMIC SAMPLING' if args.hard_threshold is not None else ''),
                  'GPU' if torch.cuda.is_available() else 'CPU',
                  "Resume from \'{}\'".format(args.resume)
                  if args.resume else "Encoder: {}".format(args.encoder),
                  args.epochs,
                  'Validate every {} epoch(s)'.format(args.test_every)
                  if args.test_every <= args.epochs else 'No validation',
                  args.lr, args.output)
          )
    print("Image batch size: {}".format(args.image_batch_size))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    training_data_path = config.get("data", "data_path")

    # data loading
    kfold = None if args.test_every > args.epochs else 10
    trainset = LystoDataset(os.path.join(training_data_path, "training.h5"), organ=args.organ, augment=args.augment,
                            kfold=kfold, shuffle=True, num_of_imgs=100 if args.debug else 0)
    trainset.setmode(5)
    collate_fn = default_collate
    # TODO: how can I split the training step for distributed parallel training?
    train_sampler = DistributedSampler(trainset) if dist.is_nccl_available() and args.distributed else None
    train_loader = DataLoader(trainset, batch_size=args.image_batch_size, shuffle=True, num_workers=args.workers,
                              sampler=train_sampler, pin_memory=True, collate_fn=collate_fn)
    if kfold is not None:
        valset = LystoDataset(os.path.join(training_data_path, "training.h5"), train=False, organ=args.organ,
                              kfold=kfold, num_of_imgs=100 if args.debug else 0)
        valset.setmode(5)
        val_sampler = DistributedSampler(valset) if dist.is_nccl_available() and args.distributed else None
        val_loader = DataLoader(valset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                                sampler=val_sampler, pin_memory=True)

    # model setup
    def to_device(model, device):
        if dist.is_nccl_available() and args.distributed:
            print('\nNCCL is available. Setup distributed parallel training with {} devices...\n'
                  .format(torch.cuda.device_count()))
            dist.init_process_group(backend='nccl', world_size=1)
            device = torch.device("cuda", args.local_rank)
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
        else:
            model.to(device)
        return model

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    model = nets[args.encoder]
    model = to_device(model, device)
    model.setmode("image")

    if args.resume:
        cp = torch.load(args.resume, map_location=device)
        last_epoch = cp['epoch']
        last_epoch_for_scheduler = cp['scheduler']['last_epoch'] if cp['scheduler'] is not None else -1
        # load params of resnet encoder and image head only
        model.load_state_dict(
            OrderedDict({k: v for k, v in cp['state_dict'].items()
                         if k.startswith(model.encoder_prefix + model.image_module_prefix)}),
            strict=False)
    else:
        last_epoch = 0
        last_epoch_for_scheduler = -1

    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.MSELoss()
    # crit_reg = WeightedMSELoss()

    # optimization settings
    optimizer_params = {'params': model.parameters(),
                        'initial_lr': args.lr}
    optimizers = {
        'SGD': optim.SGD([optimizer_params], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay),
        'Adam': optim.Adam([optimizer_params], lr=args.lr, weight_decay=args.weight_decay)
    }
    schedulers = {
        'OneCycleLR': OneCycleLR,  # note that last_epoch means last iteration number here
        'ExponentialLR': ExponentialLR,
        'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts,
    }
    scheduler_kwargs = {
        'OneCycleLR': {
            'max_lr': args.lr,  # note that input lr means max_lr here
            'epochs': args.epochs,
            'steps_per_epoch': len(train_loader),
            'div_factor': 25.0,  # initial lr = max_lr / div_factor
            'pct_start': 0.3  # percent of steps in warm-up period
        },
        'ExponentialLR': {
            'gamma': 0.9,
        },
        'CosineAnnealingWarmRestarts': {
            'T_0': 10,
        }
    }

    optimizer = optimizers['SGD'] if args.scheduler is not None else optimizers['Adam']
    # optimizer = optimizers['Adam']
    scheduler = schedulers[args.scheduler](optimizer,
                                           last_epoch=last_epoch_for_scheduler,
                                           **scheduler_kwargs[args.scheduler]) \
        if args.scheduler is not None else None
    if args.resume:
        optimizer.load_state_dict(cp['optimizer'])
        if cp['scheduler'] is not None and scheduler is not None:
            scheduler.load_state_dict(cp['scheduler'])

    if args.reg_only:
        train_reg(total_epochs=args.epochs,
                  last_epoch=last_epoch,
                  test_every=args.test_every,
                  model=model,
                  device=device,
                  crit_reg=crit_reg,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  output_path=args.output,
                  thresh=args.hard_threshold)
    else:
        train(total_epochs=args.epochs,
              last_epoch=last_epoch,
              test_every=args.test_every,
              model=model,
              device=device,
              crit_cls=crit_cls,
              crit_reg=crit_reg,
              optimizer=optimizer,
              scheduler=scheduler,
              output_path=args.output,
              thresh=args.hard_threshold)

    # train_cls(total_epochs=args.epochs,
    #           last_epoch=last_epoch,
    #           test_every=args.test_every,
    #           model=model,
    #           device=device,
    #           crit_cls=crit_cls,
    #           optimizer=optimizer,
    #           scheduler=scheduler,
    #           output_path=args.output
    #           )
    # optimizer = optimizers['SGD'] if args.scheduler is not None else optimizers['Adam']
    # scheduler = schedulers[args.scheduler](optimizer,
    #                                        last_epoch=last_epoch_for_scheduler,
    #                                        **scheduler_kwargs[args.scheduler]) \
    #     if args.scheduler is not None else None
    #
    # train_reg(total_epochs=args.epochs,
    #           last_epoch=last_epoch,
    #           test_every=args.test_every,
    #           model=model,
    #           device=device,
    #           crit_reg=crit_reg,
    #           optimizer=optimizer,
    #           scheduler=scheduler,
    #           output_path=args.output
    #           )
