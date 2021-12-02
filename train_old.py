import warnings
import os
import argparse
import time
import csv

import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import LystoDataset
from model import encoders
from inference import *
from train import *
from validation import *
from utils import default_collate

warnings.filterwarnings("ignore")
now = int(time.time())

# Training settings
parser = argparse.ArgumentParser(prog="train.py", description='Training')
parser.add_argument('-e', '--epochs', type=int, default=30,
                    help='total number of epochs to train (default: 30)')
parser.add_argument('--tile_only', action='store_true',
                    help='if whole image mode is disabled (mutually exclusive with --image_only)')
parser.add_argument('--image_only', action='store_true',
                    help='if tile mode is disabled (mutually exclusive with --tile_only)')
parser.add_argument('-E', '--encoder', type=str, default='resnet50',
                    help='structure of the shared encoder, [resnet18, resnet34, resnet50 (default)]')
parser.add_argument('-b', '--tile_batch_size', type=int, default=40960,
                    help='batch size of tiles (default: 40960)')
parser.add_argument('-B', '--image_batch_size', type=int, default=64,
                    help='batch size of images (default: 64)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-s', '--scheduler', type=str, default=None,
                    help='''learning rate scheduler if necessary, 
                    [\'OneCycleLR\', \'ExponentialLR\', \'CosineAnnealingWarmRestarts\'] (default: None)''')
parser.add_argument('-w', '--workers', default=4, type=int,
                    help='number of dataloader workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int,
                    help='validate every (default: 1) epoch(s). To use all data for training, set this greater than --epochs')
parser.add_argument('-k', '--tiles_per_pos', default=1, type=int,
                    help='k tiles are from a single positive cell (default: 1, standard MIL)')
parser.add_argument('-n', '--topk_neg', default=30, type=int,
                    help='top k tiles from a negative image (default: 30, standard MIL)')
parser.add_argument('-t', '--tile_size', type=int, default=32,
                    help='size of a certain tile (default: 32)')
parser.add_argument('-i', '--interval', type=int, default=20,
                    help='interval between adjacent tiles (default: 20)')
parser.add_argument('-c', '--threshold', type=float, default=0.88,
                    help='minimal prob for tiles to show in generating segmentation masks (default: 0.88)')
parser.add_argument('--distributed', action="store_true",
                    help='if distributed parallel training is enabled (seems to no avail)')
parser.add_argument('-d', '--device', type=int, default=0,
                    help='CUDA device id if available (default: 0, mutually exclusive with --distributed)')
parser.add_argument('-o', '--output', type=str, default='checkpoint/{}'.format(now), metavar='OUTPUT/PATH',
                    help='saving directory of output file (default: ./checkpoint/<timestamp>)')
parser.add_argument('-r', '--resume', type=str, default=None, metavar='MODEL/FILE/PATH',
                    help='continue training from a checkpoint.pth')
parser.add_argument('--local_rank', type=int, help=argparse.SUPPRESS)
args = parser.parse_args()

# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1)
# else:
#     torch.manual_seed(1)

max_acc = 0
verbose = True
now = int(time.time())

# data setup
if args.tile_only or args.image_only:
    single_branch = 'tile_only' if args.tile_only else 'image_only'
else:
    single_branch = 'tile+image'
print("Training settings: ")
print("Training Mode: {} | Device: {} | Encoder: {} | {} epoch(s) in total | {} | Initial LR: {} | Output directory: {}"
      .format(single_branch, 'GPU' if torch.cuda.is_available() else 'CPU',
              args.encoder, args.epochs,
              'Validate every {} epoch(s)'.format(args.test_every) if args.test_every <= args.epochs else 'No validation',
              args.lr, args.output))
if not args.image_only:
    print("Tile batch size: {} | Tile size: {} | Stride: {} | Negative top-k: {} |"
          .format(args.tile_batch_size, args.tile_size, args.interval, args.topk_neg))
if not args.tile_only:
    print("Image batch size: {}".format(args.image_batch_size))
if not os.path.exists(args.output):
    os.mkdir(args.output)

print('Loading Dataset ...')
kfold = None if args.test_every > args.epochs else 10
trainset = LystoDataset("data/training.h5", tile_size=args.tile_size, interval=args.interval, kfold=kfold, num_of_imgs=0)
valset = LystoDataset("data/training.h5", tile_size=args.tile_size, interval=args.interval, train=False, kfold=kfold)

collate_fn = default_collate
# TODO: how can I split the training step for distributed parallel training?
train_sampler = DistributedSampler(trainset) if dist.is_nccl_available() and args.distributed else None
val_sampler = DistributedSampler(valset) if dist.is_nccl_available() and args.distributed else None
train_loader_forward = DataLoader(trainset, batch_size=args.tile_batch_size, shuffle=True,
                                  num_workers=args.workers, sampler=train_sampler, pin_memory=True)
train_loader_backward_tile = DataLoader(trainset, batch_size=args.tile_batch_size, shuffle=True,
                                        num_workers=args.workers, sampler=train_sampler, pin_memory=True)
train_loader_backward_image = DataLoader(trainset, batch_size=args.image_batch_size, shuffle=True,
                                         num_workers=args.workers, sampler=train_sampler, pin_memory=True,
                                         collate_fn=collate_fn)
val_loader_tile = DataLoader(valset, batch_size=args.tile_batch_size, shuffle=False, num_workers=args.workers,
                             sampler=val_sampler, pin_memory=True)
val_loader_image = DataLoader(valset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.workers,
                              sampler=val_sampler, pin_memory=True)

# model setup
model = encoders[args.encoder]
# 把 ResNet 源码中的分为 1000 类改为二分类（由于预训练模型文件的限制，只能在外面改）
model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)

crit_cls = nn.CrossEntropyLoss()
crit_reg = nn.MSELoss()
# crit_reg = WeightedMSELoss()
crit_seg = None # TODO: CE?
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

optimizer_params = {'params': model.parameters(),
                    'initial_lr': args.lr}
optimizers = {
    'SGD': optim.SGD([optimizer_params], lr=args.lr, momentum=0.9, weight_decay=1e-4),
    'Adam': optim.Adam([optimizer_params], lr=args.lr, weight_decay=1e-4)
}
schedulers = {
    'OneCycleLR': OneCycleLR, # note that last_epoch means last iteration number here
    'ExponentialLR': ExponentialLR,
    'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts,
}
scheduler_kwargs = {
    'OneCycleLR': {
        'max_lr': args.lr,
        'epochs': args.epochs,
        'steps_per_epoch': len(train_loader_backward_image),
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

if args.resume:
    checkpoint = torch.load(args.resume)
    last_epoch = checkpoint['epoch']
    last_epoch_for_scheduler = checkpoint['scheduler']['last_epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])


def train(single_branch, total_epochs, last_epoch, test_every, model, crit_cls, crit_reg, crit_seg,
          optimizer, scheduler, threshold, tiles_per_pos, topk_neg, output_path):
    """one training epoch = tile mode -> image mode

    :param single_branch:   单分支训练模式，['tile_only', 'image_only', None]
    :param total_epochs:    迭代总次数
    :param last_epoch:      上一次迭代的次数（当继续训练时）
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param crit_cls:        分类损失函数
    :param crit_reg:        回归损失函数
    :param crit_seg:        分割损失函数
    :param optimizer:       优化器
    :param scheduler:       学习率调度器
    :param threshold:       验证模型所用的置信度
    :param tiles_per_pos:   在**单个阳性细胞**上选取的图像块数 (topk_pos = tiles_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k tile **总数**
    :param output_path:     保存模型文件和训练数据结果的目录
    """

    global device, now

    # open output file
    fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'w')
    fconv.write('epoch,tile_loss,image_cls_loss,image_reg_loss,image_loss,image_seg_loss\n')
    fconv.close()
    # 训练结果保存在 output_path/<timestamp>-training.csv
    if test_every <= args.epochs:
        fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'w')
        fconv.write('epoch,tile_error,tile_fpr,tile_fnr,image_map,mse,qwk\n')
        fconv.close()
    # 验证结果保存在 output_path/<timestamp>-validation.csv

    print('Training ...'if not args.resume else 'Resuming from the checkpoint (epoch {})...'.format(last_epoch))

    validate = lambda epoch, test_every: (epoch + 1) % test_every == 0
    start = int(time.time())
    with SummaryWriter() as writer:
        for epoch in range(1 + last_epoch, total_epochs + 1):
            # Training tile-mode only
            if single_branch == 'tile_only':

                # Forwarding step
                trainset.setmode(1)
                model.setmode("tile")

                probs = inference_tiles(train_loader_forward, model, device, epoch, total_epochs)
                sample(trainset, probs, tiles_per_pos, topk_neg)

                trainset.setmode(3)
                loss = train_tile(train_loader_backward_tile, epoch, total_epochs, model, device, crit_cls, optimizer, scheduler)
                print("tile loss: {:.4f}".format(loss))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                add_scalar_loss(writer, epoch, loss, ['tile'])

                # Validating step
                if validate(epoch, test_every):
                    valset.setmode(1)
                    print('Validating ...')

                    probs_t = inference_tiles(val_loader_tile, model, device, epoch, total_epochs)
                    metrics_t = validation_tile(valset, probs_t, tiles_per_pos, threshold)
                    print('tile error: {} | tile FPR: {} | tile FNR: {}'.format(*metrics_t))

                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{}\n'.format(epoch, *metrics_t))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_t, 'tile')

                save_model(epoch, model, "tile", optimizer, scheduler, output_path)

            # Training image-mode only
            elif single_branch == 'image_only':

                trainset.setmode(5)
                model.setmode("image")
                beta = 1
                gamma = 1

                loss = train_image(train_loader_backward_image, epoch, total_epochs, model, device, crit_cls,
                                   crit_reg, optimizer, scheduler, beta, gamma)

                print("image cls loss: {:.4f} | image reg loss: {:.4f} | image loss: {:.4f}"
                      .format(*loss))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},,{},{},{}\n'.format(epoch, *loss))
                fconv.close()

                add_scalar_loss(writer, epoch, loss, ['image', 'total'])

                # Validating step
                if validate(epoch, test_every):
                    print('Validating ...')

                    # image validating
                    valset.setmode(4)
                    categories, counts = inference_image(val_loader_image, model, device, epoch, total_epochs)

                    regconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
                        now, epoch)), 'w', newline="")
                    w = csv.writer(regconv, delimiter=',')
                    w.writerow(['id', 'organ', 'label', 'count', 'category', 'loss'])
                    for i, count in enumerate(counts):
                        w.writerow([i + 1, valset.organs[i], valset.labels[i], count, valset.cls_labels[i],
                                    np.abs(count - valset.labels[i])])
                    regconv.close()

                    metrics_i = validation_image(valset, categories, counts)
                    print('image categories mAP: {} | MSE: {} | QWK: {}\n'.format(*metrics_i))
                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},,,,{},{},{}\n'.format(epoch, *metrics_i))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_i, 'image')

                save_model(epoch, model, "image", optimizer, scheduler, output_path)

            # Alternative training
            else:
                # Forwarding step
                trainset.setmode(1)
                model.setmode("tile")
                probs = inference_tiles(train_loader_forward, model, device, epoch, total_epochs)
                sample(trainset, probs, tiles_per_pos, topk_neg)

                trainset.setmode(2)
                # if epoch == total_epochs:
                #     trainset.visualize_bboxes()  # tile visualize testing
                alpha = 1.
                beta = 1.
                gamma = 1.
                delta = 1.
                loss = train_alternative(train_loader_backward_image, epoch, total_epochs, model, device, crit_cls,
                                         crit_reg, crit_seg, optimizer, scheduler, threshold, alpha, beta, gamma, delta)

                print("tile loss: {:.4f} | image cls loss: {:.4f} | image reg loss: {:.4f} | image seg loss: {:.4f} | image loss: {:.4f}"
                      .format(*loss))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},{},{},{},{},{}\n'.format(epoch, *loss))
                fconv.close()

                add_scalar_loss(writer, epoch, loss, ['tile', 'image', 'seg', 'total'])

                # Validating step
                if validate(epoch, test_every):
                    valset.setmode(1)
                    model.setmode("tile")
                    print('Validating ...')

                    probs_t = inference_tiles(val_loader_tile, model, device, epoch, total_epochs)
                    metrics_t = validation_tile(valset, probs_t, tiles_per_pos, threshold)
                    print('tile error: {} | tile FPR: {} | tile FNR: {}'.format(*metrics_t))

                    # image validating
                    valset.setmode(4)
                    model.setmode("image")
                    categories, counts = inference_image(val_loader_image, model, device, epoch, total_epochs)

                    regconv = open(os.path.join(output_path, '{}-count-e{}.csv'.format(
                        now, epoch)), 'w', newline="")
                    w = csv.writer(regconv, delimiter=',')
                    w.writerow(['id', 'label', 'count', 'loss'])
                    for i, count in enumerate(counts):
                        w.writerow([i + 1, valset.labels[i], count, count - valset.labels[i]])
                    regconv.close()

                    metrics_i = validation_image(valset, categories, counts)
                    print('image categories mAP: {} | MSE: {} | QWK: {}\n'.format(*metrics_i))
                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{},{},{},{}\n'.format(epoch, *(metrics_t + metrics_i)))
                    fconv.close()

                    add_scalar_metrics(writer, epoch, metrics_t, 'tile')
                    add_scalar_metrics(writer, epoch, metrics_i, 'image')

                save_model(epoch, model, 'alternative', optimizer, scheduler, output_path)

    end = int(time.time())
    print("\nTrained for {} epochs. Model saved in \'{}\'. Runtime: {}s".format(total_epochs, output_path, end - start))


def save_model(epoch, model, mode, optimizer, scheduler, output_path):
    """用 .pth 格式保存模型。"""

    obj = {
        'mode': mode,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'encoder': model.encoder_name,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(obj, os.path.join(output_path, 'checkpoint_{}_{}epochs.pth'.format(epoch, mode)))


def add_scalar_loss(writer, epoch, losses, mode):

    losses = list(losses)

    if 'tile' in mode:
        writer.add_scalar("tile loss", losses.pop(0), epoch)

    if 'image' in mode:
        writer.add_scalar("image cls loss", losses.pop(0), epoch)
        writer.add_scalar("image reg loss", losses.pop(0), epoch)

    if 'segment' in mode:
        writer.add_scalar("image seg loss", losses.pop(0), epoch)

    if 'total' in mode:
        writer.add_scalar("image loss", losses.pop(0), epoch)


def add_scalar_metrics(writer, epoch, metrics, mode):

    metrics = list(metrics)

    if mode == 'tile':
        assert len(metrics) == 3, "Tile metrics should include 3 items: error rate, FPR and FNR. "
        writer.add_scalar('tile error rate', metrics[0], epoch)
        writer.add_scalar('tile false positive rate', metrics[1], epoch)
        writer.add_scalar('tile false negative rate', metrics[2], epoch)

    else: # mode == 'image'
        assert len(metrics) == 3, "Image metrics should include 3 items: mAP, MSE and QWK. "
        writer.add_scalar('image map', metrics[0], epoch)
        writer.add_scalar('image mse', metrics[1], epoch)
        writer.add_scalar('image qwk', metrics[2], epoch)


if __name__ == "__main__":

    train(single_branch=single_branch,
          total_epochs=args.epochs,
          last_epoch=last_epoch,
          test_every=args.test_every,
          model=model,
          crit_cls=crit_cls,
          crit_reg=crit_reg,
          crit_seg=crit_seg,
          optimizer=optimizer,
          scheduler=scheduler,
          threshold=args.threshold,
          tiles_per_pos=args.tiles_per_pos,
          topk_neg=args.topk_neg,
          output_path=args.output
          )
