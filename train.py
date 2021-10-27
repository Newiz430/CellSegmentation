import warnings
import os
import numpy as np
import argparse
from tqdm import tqdm
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import model.resnet as models
from utils.collate import default_collate

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(prog="train.py", description='Training')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-P', '--patch_only', action='store_true', help='if slide mode is disabled')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='mini-batch size of images (default: 32)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-t', '--test_every', default=1, type=int, help='test on val every (default: 1)')
parser.add_argument('-p', '--patches_per_pos', default=1, type=int,
                    help='k tiles are from a single positive cell (default: 1, standard MIL)')
parser.add_argument('-n', '--topk_neg', default=30, type=int,
                    help='top k tiles from a negative slide (default: 30, standard MIL)')
parser.add_argument('--patch_size', type=int, default=32, help='size of each slide (default: 32)')
parser.add_argument('-d', '--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('-o', '--output', type=str, default='.', help='name of output file')
parser.add_argument('-r', '--resume', action='store_true', help='continue training from a checkpoint file.pth')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

max_acc = 0
resume = False
verbose = True
now = int(time.time())

trainset = None
valset = None

model = models.MILresnet18(pretrained=True)

if args.resume:
    resume = True
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

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
model.to(device)

def train(batch_size, patch_only, workers, total_epochs, test_every, model,
          crit_cls, crit_reg, crit_seg, optimizer, patches_per_pos, topk_neg, output_path):
    """one training epoch = patch mode -> slide mode

    :param batch_size:      DataLoader 打包的小 batch 大小
    :param workers:         DataLoader 使用的进程数
    :param total_epochs:    迭代总次数
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param crit_cls:        分类器损失函数
    :param crit_reg:        回归损失函数
    :param crit_seg:        分割损失函数
    :param optimizer:       优化器
    :param patches_per_pos: 在**单个阳性细胞**上选取的 patch 数 (topk_pos = patches_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k patch **总数**
    :param output_path:     保存模型文件和训练数据结果的目录
    """

    global device, resume, now

    # shuffle 只能是 False
    # 暂定对 patch 的训练和对 slide 的训练所用的 batch_size 是一样的
    collate_fn = default_collate
    train_loader_forward = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                      pin_memory=True)
    train_loader_backward = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True)

    # open output file
    fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'w')
    fconv.write('epoch,patch_loss,slide_cls_loss,slide_reg_loss,slide_seg_loss,total_loss\n')
    fconv.close()
    # 训练结果保存在 output_path/<timestamp>-training.csv
    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'w')
    fconv.write('epoch,patch_error,patch_fpr,patch_fnr,slide_error,slide_fpr,slide_fnr,mae,mse\n')
    fconv.close()
    # 验证结果保存在 output_path/<timestamp>-validation.csv

    print('Start training ...')
    # if resume:
    #     print('Resuming from the checkpoint (epochs: {}).'.format(model['epoch']))

    with SummaryWriter() as writer:
        for epoch in range(1, total_epochs + 1):
            start = time.time()

            # Forwarding step
            # 把 ResNet 源码中的分为 1000 类改为二分类（由于预训练模型文件的限制，只能在外面改）
            model.fc_patch = nn.Linear(model.fc_patch.in_features, 2).to(device)
            trainset.setmode(1)
            probs = predict_patch(train_loader_forward, batch_size, epoch, total_epochs)
            sample(probs, patches_per_pos, topk_neg)

            # Training patch-mode only
            if patch_only:
                trainset.setmode(3)
                loss = train_patch(train_loader_forward, batch_size, epoch, total_epochs, model, crit_cls, optimizer)

                end = time.time()

                print("patch loss: {:.4f}".format(loss))
                print("Runtime: {}s".format((end - start) / 1000))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},{}\n'.format(epoch, loss))
                fconv.close()

                writer.add_scalar("patch loss", loss, epoch)

                # Validating step
                if (epoch + 1) % test_every == 0:
                    valset.setmode(1)
                    print('Validating ...')

                    probs_p = predict_patch(val_loader, batch_size, epoch, total_epochs)
                    metrics_p = validation_patch(probs_p)
                    print('patch error: {} | patch FPR: {} | patch FNR: {}'.format(*metrics_p))

                    writer.add_scalar('patch error rate', metrics_p[0], epoch)
                    writer.add_scalar('patch false positive rate', metrics_p[1], epoch)
                    writer.add_scalar('patch false negative rate', metrics_p[2], epoch)
                    
                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{}\n'.format(epoch, *metrics_p))
                    fconv.close()

                    # 每验证一次，保存模型
                    obj = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(output_path, 'checkpoint_{}epochs_patchonly.pth'.format(epoch)))

            # Alternative training step
            else:
                trainset.setmode(2)
                if epoch == total_epochs:
                    trainset.visualize_bboxes()  # patch visualize testing
                alpha = 1.
                beta = 0.1
                gamma = 0.1
                delta = 0.1
                loss = train_alternative(train_loader_backward, batch_size, epoch, total_epochs, model, crit_cls,
                                         crit_reg, crit_seg, optimizer, alpha, beta, gamma, delta)

                end = time.time()

                print("patch loss: {:.4f} | slide cls loss: {:.4f} | slide reg loss: {:.4f} | slide seg loss: {:.4f} | slide loss: {:.4f}"
                      .format(*loss))
                print("Runtime: {}s".format((end - start) / 1000))
                fconv = open(os.path.join(output_path, '{}-training.csv'.format(now)), 'a')
                fconv.write('{},{},{},{},{},{}\n'.format(epoch, *loss))
                fconv.close()

                writer.add_scalar("patch loss", loss[0], epoch)
                writer.add_scalar("slide cls loss", loss[1], epoch)
                writer.add_scalar("slide reg loss", loss[2], epoch)
                writer.add_scalar("slide seg loss", loss[3], epoch)
                writer.add_scalar("slide loss", loss[4], epoch)

                # Validating step
                if (epoch + 1) % test_every == 0:
                    valset.setmode(1)
                    print('Validating ...')

                    probs_p = predict_patch(val_loader, batch_size, epoch, total_epochs)
                    metrics_p = validation_patch(probs_p)
                    print('patch error: {} | patch FPR: {} | patch FNR: {}'.format(*metrics_p))

                    writer.add_scalar('patch error rate', metrics_p[0], epoch)
                    writer.add_scalar('patch false positive rate', metrics_p[1], epoch)
                    writer.add_scalar('patch false negative rate', metrics_p[2], epoch)

                    # slide validating
                    valset.setmode(4)
                    probs_s, reg, seg = predict_slide(val_loader, batch_size, epoch, total_epochs)
                    metrics_s = validation_slide(probs_s, reg, seg)
                    print('slide error: {} | slide FPR: {} | slide FNR: {}\nMAE: {} | MSE: {}\n'.format(*metrics_s))
                    fconv = open(os.path.join(output_path, '{}-validation.csv'.format(now)), 'a')
                    fconv.write('{},{},{},{},{},{},{},{},{}\n'.format(epoch, *(metrics_p + metrics_s)))
                    fconv.close()

                    writer.add_scalar('slide error rate', metrics_s[0], epoch)
                    writer.add_scalar('slide false positive rate', metrics_s[1], epoch)
                    writer.add_scalar('slide false negative rate', metrics_s[2], epoch)
                    writer.add_scalar('slide mae', metrics_s[3], epoch)
                    writer.add_scalar('slide mse', metrics_s[4], epoch)

                    # 每验证一次，保存模型
                    obj = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(obj, os.path.join(output_path, 'checkpoint_{}epochs.pth'.format(epoch)))


def predict_patch(loader, batch_size, epoch, total_epochs):
    """前馈推导一次模型，获取实例分类概率。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    """
    global device

    model.setmode("patch")
    model.eval()

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        patch_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, input in enumerate(patch_bar):
            patch_bar.set_postfix(step="patch forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            # softmax 输出 [[a,b],[c,d]] shape = batch_size*2
            output = model(input[0].to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] 取出 softmax 得到的概率，产生：[b, d, ...]
            # input.size(0) 返回 batch 中的实例数量
            probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def sample(probs, patches_per_pos, topk_neg):
    """找出概率为 top-k 的补丁，制作迭代使用的数据集。

    :param probs:           predict_patch() 得到的补丁概率
    :param patches_per_pos: 在**单个阳性细胞**上选取的 patch 数 (topk_pos = patches_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k patch **总数**
    """

    global verbose

    groups = np.array(trainset.patchIDX)
    order = np.lexsort((probs, groups))

    index = np.empty(len(trainset), 'bool')
    for i in range(len(trainset)):
        topk = topk_neg if trainset.labels[groups[i]] == 0 else trainset.labels[groups[i]] * patches_per_pos
        index[i] = groups[i] != groups[(i + topk) % len(groups)]

    p, n = trainset.make_train_data(list(order[index]))
    if verbose:
        print("Training data is sampled. \nPos samples: {} | Neg samples: {}".format(p, n))


def predict_slide(loader, batch_size, epoch, total_epochs):
    """前馈推导一次模型，获取图像级的分类概率和回归预测值。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :return:                切片分类概率，细胞计数，分割结果
    """

    model.setmode("slide")
    model.eval()

    probs = torch.tensor(())
    nums = torch.tensor(())
    feats = torch.tensor(())
    with torch.no_grad():
        slide_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, (data, label_cls, label_num, _) in enumerate(slide_bar):
            slide_bar.set_postfix(step="slide forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            output = model(data.to(device))
            output_cls = F.softmax(output[0], dim=1)
            probs = torch.cat((probs, output_cls.detach()[:, 1].clone().cpu()), dim=0)
            nums = torch.cat((nums, output[1].detach()[:, 0].clone().cpu()), dim=0)
            feats = torch.cat((feats, output[2].detach().clone().cpu()), dim=0)
    return probs.numpy(), nums.numpy(), feats.numpy()


def train_patch(loader, batch_size, epoch, total_epochs, model, criterion, optimizer):
    """Patch training for one epoch.

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

    patch_num = 0
    train_loss = 0.
    train_bar = tqdm(loader, total=len(loader))
    for i, (data, label) in enumerate(train_bar):
        train_bar.set_postfix(step="patch training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))

        output = model(data.to(device))
        optimizer.zero_grad()
        loss = criterion(output, label.to(device))

        patch_num += data.size(0)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    train_loss /= patch_num
    return train_loss


def train_alternative(loader, batch_size, epoch, total_epochs, model, crit_cls, crit_reg, crit_seg, optimizer, alpha, beta, gamma, delta):
    """patch + slide training for one epoch. slide mode = slide_cls + slide_reg + slide_seg

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param crit_cls:        分类器损失函数
    :param crit_reg:        回归损失函数
    :param crit_seg:        分割损失函数
    :param optimizer:       优化器
    :param alpha:           patch_loss 系数
    :param beta:            slide_cls_loss 系数
    :param gamma:           slide_reg_loss 系数
    :param delta:           slide_seg_loss 系数
    """

    global device

    model.train()

    patch_num = 0
    patch_loss = 0.
    slide_cls_loss = 0.
    slide_reg_loss = 0.
    slide_seg_loss = 0.
    slide_loss = 0.

    train_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
    for i, (data, labels) in enumerate(train_bar):
        train_bar.set_postfix(step="alternative training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))

        # Patch training
        model.setmode("patch")
        # print("slides pack size:", data[0].size())
        # print("patches pack size:", data[1].size())
        output = model(data[1].to(device))
        optimizer.zero_grad()

        patch_loss_i = alpha * crit_cls(output, labels[3].to(device))
        patch_loss_i.backward()
        optimizer.step()

        patch_loss += patch_loss_i.item() * data[1].size(0)
        patch_num += data[1].size(0)

        # Slide training
        model.setmode("slide")

        output = model(data[0].to(device))

        optimizer.zero_grad()

        slide_cls_loss_i = crit_cls(output[0], labels[0].to(device))
        slide_reg_loss_i = crit_reg(output[1].squeeze(), labels[1].to(device, dtype=torch.float32))
        # slide_seg_loss_i = crit_seg(output[2], labels[2].to(device))

        # total_loss_i = alpha * patch_loss_i + beta * slide_cls_loss_i + \
        #                gamma * slide_reg_loss_i + delta * slide_seg_loss_i
        slide_loss_i = beta * slide_cls_loss_i + gamma * slide_reg_loss_i
        slide_loss_i.backward()
        optimizer.step()

        slide_cls_loss += slide_cls_loss_i.item() * data[0].size(0)
        slide_reg_loss += slide_reg_loss_i.item() * data[0].size(0)
        # slide_seg_loss += slide_seg_loss_i.item() * slide_data[0].size(0)
        slide_loss += slide_loss_i.item() * data[0].size(0)

        # print("slide data size:", data[0].size(0))
        # print("patch data size:", data[1].size(0))

    # print("Total patches:", patch_num)
    # print("Total slides:", len(loader.dataset))

    patch_loss /= patch_num
    slide_loss /= len(loader.dataset)
    slide_cls_loss /= len(loader.dataset)
    slide_reg_loss /= len(loader.dataset)
    # slide_seg_loss /= len(loader.dataset)
    slide_seg_loss = 0.
    return patch_loss, slide_cls_loss, slide_reg_loss, slide_seg_loss, slide_loss

# def validation_patch(probs):
#     """patch mode 的验证"""
#
#     val_groups = np.array(valset.patchIDX)
#
#     max_prob = np.empty(len(valset.labels))  # 模型预测的实例最大概率列表，每张切片取最大概率的 patch
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
#     probs = np.round(max_prob)  # 每张切片由最大概率的 patch 得到的标签
#     err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
#     return err, fpr, fnr

def validation_patch(probs):
    """patch mode 的验证"""

    thr = 0.88  # patch 的分类阈值，根据经验设定
    val_groups = np.array(valset.patchIDX)

    order = np.lexsort((probs, val_groups))
    val_groups = val_groups[order]
    val_probs = probs[order]

    val_index = np.empty(len(val_probs), 'bool')
    val_index = np.array([prob > thr for prob in val_probs])

    # 制作分类用的 label
    labels = np.zeros(len(val_probs))
    for i in range(1, len(val_probs) + 1):
        if i == len(val_probs) or val_groups[i] != val_groups[i - 1]:
            labels[i - valset.labels[val_groups[i - 1]]: i] = [1] * valset.labels[val_groups[i - 1]]

    # 计算错误率、FPR、FNR
    err, fpr, fnr = calc_err(val_index, labels)
    return err, fpr, fnr

def validation_slide(probs, reg, seg):
    """slide mode 的验证"""

    probs = np.round(probs)
    err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
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
    print("Training Mode: {} | Epochs: {} | Validate every {} iteration(s) | Slide batch size: {} | Negative top-k: {}"
          .format("patch" if args.patch_only else "patch + slide", args.epochs, args.test_every, args.batch_size, args.topk_neg))

    print('Loading Dataset ...')
    trainset = LystoDataset(filepath="data/training.h5", transform=trans)
    valset = LystoDataset(filepath="data/training.h5", train=False, transform=trans)

    train(batch_size=args.batch_size,
          patch_only=args.patch_only,
          workers=args.workers,
          total_epochs=args.epochs,
          test_every=args.test_every,
          model=model,
          crit_cls=crit_cls,
          crit_reg=crit_reg,
          crit_seg=crit_seg,
          optimizer=optimizer,
          patches_per_pos=args.patches_per_pos,
          topk_neg=args.topk_neg,
          output_path=args.output)
