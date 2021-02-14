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
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--topk', default=1, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
args = parser.parse_args()

torch.manual_seed(1)

print('Init Model ...')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
# trans = transforms.Compose([transforms.ToTensor(), normalize])
trans = transforms.ToTensor()  # TODO: 设计归一化方式

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

print('Loading Dataset ...')

imageSet = LystoDataset(filepath="D:/LYSTO/training.h5", transform=trans)
train_loader = DataLoader(imageSet, batch_size=2, shuffle=False)


def train(loader, model, criterion, optimizer, total_epochs, topk=1):
    print('Start training ...')

    for epoch in range(1, total_epochs + 1):

        imageSet.setmode(1)

        # 获取实例分类概率
        model.eval()
        probs = torch.FloatTensor(len(loader.dataset))
        with torch.no_grad():
            # 禁止反向传播
            for i, input in enumerate(loader):
                print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(epoch, total_epochs, i + 1, len(loader)))
                # softmax 输出[[a,b],[c,d]] shape = batch_size*2
                output = F.softmax(model(input[0]), dim=1)
                # detach()[:,1]取出softmax得到的概率，产生：[b, d, ...]
                # detach()和clone()把数据从计算图中脱离，脱离出来的新tensor和原tensor在梯度或数据上没有任何关系
                # input.size(0)返回batch中的实例数量
                probs[i * args.batch_size:i * args.batch_size + input[0].size(0)] = output.detach()[:, 1].clone()

        # 找出top-k
        probs = probs.numpy()
        groups = np.array(imageSet.imageIDX)
        order = np.lexsort((probs, groups))
        groups = groups[order]
        # probs = probs[order]
        index = np.empty(len(groups), 'bool')
        index[-topk:] = True
        # 同时把属于每个slide的、pred最大的k个实例挑出来，放入topk中
        index[:-topk] = groups[topk:] != groups[:-topk]

        # 根据top-k的分类，制作迭代使用的数据集
        imageSet.make_train_data(list(order[index]))

        imageSet.setmode(2)

        # 训练
        model.train()
        train_loss = 0.
        for i, (data, label) in enumerate(loader):
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
        train_loss /= len(loader)
        print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, total_epochs, train_loss))


if __name__ == "__main__":
    train(train_loader, model, criterion, optimizer, args.epochs, args.topk)
