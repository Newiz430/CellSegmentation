import os
import numpy as np
import argparse

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from datasets import LystoDataset

# Training settings
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size (default: 2)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--topk', default=1, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument('--output', type=str, default='.', help='name of output file')
# TODO: 分段训练
# parser.add_argument('--resume', type=str, default=None, help='continue training from a checkpoint')
args = parser.parse_args()

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nCUDA is available.\n')
# TODO: GPU训练

max_acc = 0

print('Init Model ...')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
trans = transforms.Compose([transforms.ToTensor(), normalize])  # TODO: 设计归一化方式

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

print('Loading Dataset ...')

imageSet = LystoDataset(filepath="D:/LYSTO/training.h5", transform=trans)
imageSet_val = LystoDataset(filepath="D:/LYSTO/training.h5", transform=trans, train=False)


def train(trainset, valset, batch_size, total_epochs,
          test_every, model, criterion, optimizer,
          topk, output_path):
    """
    :param trainset:        训练数据集
    :param valset:          验证数据集
    :param batch_size:      Dataloader 打包的小 batch 大小
    :param total_epochs:    迭代次数
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param criterion:       损失函数
    :param optimizer:       优化器
    :param topk:            每次选取的 top-k 实例个数
    :param output_path:     保存模型文件的目录
    """

    global max_acc

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print('Start training ...')

    for epoch in range(1, total_epochs + 1):

        trainset.setmode(1)

        # 获取实例分类概率
        model.eval()
        probs = torch.FloatTensor(len(train_loader.dataset))
        with torch.no_grad():
            # 禁止反向传播
            for i, input in enumerate(train_loader):
                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'
                      .format(epoch, total_epochs, i + 1, len(train_loader)))
                # softmax 输出[[a,b],[c,d]] shape = batch_size*2
                output = F.softmax(model(input[0]), dim=1)
                # detach()[:,1]取出softmax得到的概率，产生：[b, d, ...]
                # input.size(0)返回batch中的实例数量
                probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()

        # 找出top-k
        probs = probs.numpy()
        groups = np.array(trainset.imageIDX)
        order = np.lexsort((probs, groups))
        groups = groups[order]
        # probs = probs[order]
        index = np.empty(len(groups), 'bool')
        index[-topk:] = True
        # 同时把属于每个slide的、pred最大的k个实例挑出来，放入topk中
        index[:-topk] = groups[topk:] != groups[:-topk]

        # 根据top-k的分类，制作迭代使用的数据集
        trainset.make_train_data(list(order[index]))

        trainset.setmode(2)

        # 训练
        model.train()
        train_loss = 0.
        for i, (data, label) in enumerate(train_loader):
            output = model(data)
            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = criterion(output, label)
            train_loss += loss.item() * data.size(0)

            # backward pass
            loss.backward()
            # step
            optimizer.step()

        # calculate loss and error for epoch
        train_loss /= len(train_loader)
        print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, total_epochs, train_loss))

        # 验证

        if (epoch + 1) % test_every == 0:

            print('\nValidating ...')

            valset.setmode(1)
            model.eval()
            val_probs = torch.FloatTensor(len(val_loader.dataset))
            with torch.no_grad():
                # 禁止反向传播
                for i, input in enumerate(val_loader):
                    print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'
                          .format(epoch, total_epochs, i + 1, len(val_loader)))
                    # 把训练过的 model 在验证集上做一下
                    val_output = F.softmax(model(input[0]), dim=1)
                    val_probs[i * batch_size:i * batch_size + input[0].size(0)] \
                        = val_output.detach()[:, 1].clone()

            val_probs = val_probs.numpy()
            val_groups = np.array(valset.imageIDX)

            # TODO: 暂时把标签当作非计数式标签处理
            max_prob = np.empty(len(valset.labels))  # 模型预测的实例最大概率列表，每张切片取最大概率的 patch
            max_prob[:] = np.nan
            order = np.lexsort((val_probs, val_groups))
            # 排序
            val_groups = val_groups[order]
            val_probs = val_probs[order]
            # 取最大
            val_index = np.empty(len(val_groups), 'bool')
            val_index[-1] = True
            val_index[:-1] = val_groups[1:] != val_groups[:-1]
            max_prob[val_groups[val_index]] = val_probs[val_index]

            pred = [1 if prob >= 0.5 else 0 for prob in max_prob] # 每张切片由最大概率的 patch 得到的标签
            err, fpr, fnr = calc_err(pred, valset.labels)
            # 计算
            print('Validation\tEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}\n'
                  .format(epoch, total_epochs, err, fpr, fnr))

            # Save the best model
            acc = 1 - (fpr + fnr) / 2.
            if acc >= max_acc:
                max_acc = acc
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'max_accuracy': max_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(output_path, 'checkpoint_best.pth'))

def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr


if __name__ == "__main__":

    train(imageSet, imageSet_val, args.batch_size, args.epochs, args.test_every, model, criterion, optimizer,
          args.topk, args.output)
