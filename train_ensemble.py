import warnings
import os
import sys
import configparser
import argparse
import time
import csv
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter

from dataset import EnsembleSet
from model import nets
from evaluate import evaluate_image
from utils import default_collate

warnings.filterwarnings("ignore")
now = int(time.time())

# Training settings
parser = argparse.ArgumentParser(prog="train_ensemble.py", description='pt.1: image assessment training.')
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='total number of epochs to train (default: 30)')
parser.add_argument('-E', '--encoder', type=str, default='resnet50',
                    help='structure of the shared encoder, {\'resnet18\', \'resnet34\', \'resnet50\' (default), '
                         '\'efficientnet_b0\', \'efficientnet_b2\'}')
parser.add_argument('-k', '--kfold', type=int, default=10,
                    help='number of base models (must be at least 2, default: 10)')
parser.add_argument('-B', '--image_batch_size', type=int, default=48,
                    help='batch size of images (default: 48)')
parser.add_argument('-l', '--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='decay of weight (default: 1e-4)')
parser.add_argument('-s', '--scheduler', type=str, default=None,
                    help='learning rate scheduler if necessary, '
                         '{\'ExponentialLR\', \'CosineAnnealingWarmRestarts\'} (default: None)')
parser.add_argument('-a', '--augment', action="store_true", help='apply data augmentation')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int,
                    help='validate every (default: 1) epoch(s). ')
parser.add_argument('--distributed', action="store_true",
                    help='if distributed parallel training is enabled (seems to be no avail)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint/{}'.format(now), metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint/<timestamp>)')
parser.add_argument('--debug', action='store_true', help='use little data for debugging')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()


def train_ensemble(total_epochs, idx, last_epoch, test_every, model, device, crit_reg, optimizer, scheduler, output_path):
    from train import train_image_reg
    from inference import inference_image_reg

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

                data.setmode(True, 5)
                train_loader = data.get_loader(True, idx, batch_size=args.image_batch_size, num_workers=args.workers,
                                               collate_fn=collate_fn)
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
                    data.setmode(False, 4)
                    val_loader = data.get_loader(False, idx, batch_size=args.image_batch_size,
                                                 num_workers=args.workers)
                    counts = inference_image_reg(val_loader, model, device, epoch, total_epochs)

                    regconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
                        now, epoch)), 'w', newline="")
                    w = csv.writer(regconv, delimiter=',')
                    w.writerow(['id', 'organ', 'label', 'count', 'category label', 'loss'])
                    for i, count in enumerate(np.round(counts).astype(int)):
                        w.writerow([i + 1, data.validating_set.organs[i], data.validating_set.labels[i],
                                    count, data.validating_set.cls_labels[i],
                                    np.abs(count - data.validating_set.labels[i])])
                    regconv.close()

                    metrics_i = evaluate_image(data.validating_set, [], counts)
                    print('image categories mAP: {} | MSE: {} | QWK: {}\n'.format(*metrics_i))
                    fconv = open(os.path.join(output_path, '{}-image-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{}\n'.format(epoch, *metrics_i[1:]))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_i)

                save_model(epoch, model, optimizer, scheduler, output_path, prefix='reg_pt1_{}'.format(idx))

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path, prefix='reg_pt1_{}'.format(idx))
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
          .format('tile + image (pt.1, ENSEMBLE WITH FOLD = {})'.format(args.kfold),
                  'GPU' if torch.cuda.is_available() else 'CPU',
                  "Encoder: {}".format(args.encoder), args.epochs,
                  'Validate every {} epoch(s)'.format(args.test_every)
                      if args.test_every <= args.epochs else 'No validation',
                  args.lr, args.output)
          )
    print("Image batch size: {}".format(args.image_batch_size))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    training_data_path = config.get("data", "data_path")

    # data loading
    assert args.kfold >= 2, "K-fold cross-validation requires kfold >= 2. "
    data = EnsembleSet(os.path.join(training_data_path, "training.h5"), augment=args.augment, k=args.kfold)

    collate_fn = default_collate
    data.setmode(True, 5)
    data.setmode(False, 5)

    # model setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    models = [deepcopy(nets[args.encoder]) for _ in range(args.kfold)]

    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.MSELoss()
    # crit_reg = WeightedMSELoss()

    for i in range(args.kfold):

        print('Training {}/{}...'.format(i + 1, args.kfold))
        model = models[i].to(device)
        model.setmode("image")

        last_epoch = 0
        last_epoch_for_scheduler = -1

        # optimization settings
        optimizer_params = {'params': model.parameters(),
                            'initial_lr': args.lr}
        optimizers = {
            'SGD': optim.SGD([optimizer_params], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay),
            'Adam': optim.Adam([optimizer_params], lr=args.lr, weight_decay=args.weight_decay)
        }
        schedulers = {
            'ExponentialLR': ExponentialLR,
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts,
        }
        scheduler_kwargs = {
            'ExponentialLR': {
                'gamma': 0.9,
            },
            'CosineAnnealingWarmRestarts': {
                'T_0': 10,
            }
        }

        # optimizer = optimizers['SGD'] if args.scheduler is not None else optimizers['Adam']
        optimizer = optimizers['Adam']
        scheduler = schedulers[args.scheduler](optimizer,
                                               last_epoch=last_epoch_for_scheduler,
                                               **scheduler_kwargs[args.scheduler]) \
            if args.scheduler is not None else None

        train_ensemble(total_epochs=args.epochs,
                       idx=i,
                       last_epoch=last_epoch,
                       test_every=args.test_every,
                       model=model,
                       device=device,
                       crit_reg=crit_reg,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       output_path=args.output
                       )
