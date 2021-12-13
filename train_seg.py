import warnings
import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import LystoDataset, Maskset
from model import encoders
from inference import inference_tiles
from train import train_seg
from utils import generate_masks

warnings.filterwarnings("ignore")
now = int(time.time())

# Training settings
parser = argparse.ArgumentParser(prog="train_seg.py", description='pt.3: cell segmentation branch training.')
parser.add_argument('-m', '--model', type=str, help='path to pretrained model in pt.1')
parser.add_argument('--skip_draw', action='store_true',
                    help='skip binary mask generating step, using the images from data/pseudomask instead')
parser.add_argument('-b', '--tile_batch_size', type=int, default=40960,
                    help='batch size of tiles (useless in --skip_draw mode, default: 40960)')
parser.add_argument('-i', '--interval', type=int, default=5,
                    help='sample interval of tiles (default: 5)')
parser.add_argument('-p', '--tile_size', type=int, default=16,
                    help='size of each tile (default: 16)')
parser.add_argument('-B', '--image_batch_size', type=int, default=64,
                    help='batch size of images (default: 64)')
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='total number of epochs to train (default: 30)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-s', '--scheduler', type=str, default=None,
                    help='learning rate scheduler if necessary, '
                         '[\'OneCycleLR\', \'ExponentialLR\', \'CosineAnnealingWarmRestarts\'] (default: None)')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int,
                    help='validate every (default: 1) epoch(s). To use all data for training, '
                         'set this greater than --epochs')
parser.add_argument('-c', '--threshold', type=float, default=0.95,
                    help='minimal prob for tiles to show in generating segmentation masks (default: 0.95)')
parser.add_argument('--distributed', action="store_true",
                    help='if distributed parallel training is enabled (seems to no avail)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint/{}'.format(now), metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint/<timestamp>)')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()


def train(total_epochs, last_epoch, test_every, model, device, optimizer, scheduler, output_path):
    """pt.2: tile classifier training.

    :param total_epochs:    迭代总次数
    :param last_epoch:      上一次迭代的次数（当继续训练时）
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param device:          模型所在的设备
    :param optimizer:       优化器
    :param scheduler:       学习率调度器
    :param output_path:     保存模型文件和训练数据结果的目录
    """

    fconv = open(os.path.join(output_path, '{}-seg-training.csv'.format(now)), 'w')
    fconv.write('epoch,image_cls_loss,image_reg_loss,image_loss,image_seg_loss\n')
    fconv.close()

    print('Training ...' if not args.resume else 'Resuming from the checkpoint (epoch {})...'.format(last_epoch))

    validate = lambda epoch, test_every: (epoch + 1) % test_every == 0
    start = int(time.time())
    with SummaryWriter() as writer:
        delta = 1

        print("PT.III - cell segmentation branch training ...")
        model.setmode("segment")

        for epoch in range(1 + last_epoch, total_epochs + 1):
            try:
                if device.type == 'cuda':
                    torch.cuda.manual_seed(epoch)
                else:
                    torch.manual_seed(epoch)

                loss = train_seg(train_loader, epoch, total_epochs, model, device, optimizer,
                                 scheduler, delta)

                print("image seg loss: {:.4f}".format(loss))
                fconv = open(os.path.join(output_path, '{}-seg-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                add_scalar_loss(writer, epoch, loss)

                # # Validating step
                # if validate(epoch, test_every):
                #     print('Validating ...')
                #
                #     valset.setmode()

                save_model(epoch, model, optimizer, scheduler, output_path)

            except KeyboardInterrupt:
                save_model(epoch, model, optimizer, scheduler, output_path)
                print("\nTraining interrupted at epoch {}. Model saved in \'{}\'.".format(epoch, output_path))
                sys.exit(0)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def rank(dataset, probs, threshold):
    """按概率对 tile 排序，便于与置信度进行比较。"""

    groups = np.array(dataset.tileIDX)
    tiles = np.array(dataset.tiles_grid)

    order = np.lexsort((probs, groups))
    groups = groups[order]
    probs = probs[order]
    tiles = tiles[order]

    # index = np.empty(len(groups), 'bool')
    # index[-topk:] = True
    # index[:-topk] = groups[topk:] != groups[:-topk]
    index = [prob > threshold for prob in probs]

    return tiles[index], probs[index], groups[index]


def save_model(epoch, model, optimizer, scheduler, output_path):
    """用 .pth 格式保存模型。"""

    obj = {
        'mode': 'seg',
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'encoder': model.encoder_name,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(obj, os.path.join(output_path, 'pt3_{}epochs.pth'.format(epoch)))


def add_scalar_loss(writer, epoch, loss):
    writer.add_scalar("image seg loss", loss, epoch)


if __name__ == "__main__":

    print("Training settings: ")
    print("Training Mode: {} | Device: {} | Model: {} | {} epoch(s) in total | "
          "{} | Initial LR: {} | Output directory: {}"
          .format('tile + image (pt.3)', 'GPU' if torch.cuda.is_available() else 'CPU',
                  args.model, args.epochs,
                  'Validate every {} epoch(s)'.format(
                      args.test_every) if args.test_every <= args.epochs else 'No validation',
                  args.lr, args.output))
    print("Tile batch size: {} | Tile size: {} | Stride: {} | Image batch size: {}"
          .format(args.tile_batch_size, args.tile_size, args.interval, args.image_batch_size))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    print('Generating masks using the pretrained model \'{}\' ...'.format(args.model))

    f = torch.load(args.model)
    model = encoders[f['encoder']]
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    epoch = f['epoch']
    model.load_state_dict(f['state_dict'])
    # freeze resnet encoder
    model.set_resnet_module_grads(False)

    last_epoch = 0
    last_epoch_for_scheduler = -1

    if dist.is_nccl_available() and args.distributed:
        print('\nNCCL is available. Setup distributed parallel training with {} devices...\n'
              .format(torch.cuda.device_count()))
        dist.init_process_group(backend='nccl', world_size=1)
        device = torch.device("cuda", args.local_rank)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
        model.to(device)

    kfold = None if args.test_every > args.epochs else 10
    if not args.skip_draw:

        dataset = LystoDataset("data/training.h5", tile_size=args.tile_size, interval=args.interval, augment=False,
                               kfold=None, num_of_imgs=100)
        loader = DataLoader(dataset, batch_size=args.tile_batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=False)
        dataset.setmode(1)
        model.setmode("tile")

        probs = inference_tiles(loader, model, device)
        tiles, _, groups = rank(dataset, probs, args.threshold)
        pseudo_masks = generate_masks(dataset, tiles, groups)

        trainset = Maskset("data/training.h5", pseudo_masks)

    else:
        trainset = Maskset("data/training.h5", "data/pseudomask")

    train_sampler = DistributedSampler(trainset) if dist.is_nccl_available() and args.distributed else None
    train_loader = DataLoader(trainset, batch_size=args.image_batch_size, shuffle=True, num_workers=args.workers,
                              sampler=train_sampler, pin_memory=True)

    optimizer_params = {'params': model.parameters(),
                        'initial_lr': args.lr}
    optimizers = {
        'SGD': optim.SGD([optimizer_params], lr=args.lr, momentum=0.9, weight_decay=1e-4),
        'Adam': optim.Adam([optimizer_params], lr=args.lr, weight_decay=1e-4)
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
        },
        'ExponentialLR': {
            'gamma': 0.9,
        },
        'CosineAnnealingWarmRestarts': {
            'T_0': 5,
        }
    }

    optimizer = optimizers['SGD'] if args.scheduler is not None else optimizers['Adam']
    scheduler = schedulers[args.scheduler](optimizer,
                                           last_epoch=last_epoch_for_scheduler,
                                           **scheduler_kwargs[args.scheduler]) \
        if args.scheduler is not None else None

    train(total_epochs=args.epochs,
          last_epoch=last_epoch,
          test_every=args.test_every,
          model=model,
          device=device,
          optimizer=optimizer,
          scheduler=scheduler,
          output_path=args.output
          )
