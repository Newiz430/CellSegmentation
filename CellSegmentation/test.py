import os
import numpy as np
import argparse
from tqdm import tqdm
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Testing & Heatmap')
parser.add_argument('--model', type=str, default='checkpoint_best.pth', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size (default: 2)')
parser.add_argument('--topk', default=1, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
parser.add_argument('--interval', type=int, default=10, help='sample interval of patches (default: 10)')
parser.add_argument('--patch_size', type=int, default=32, help='size of each patch (default: 32)')
parser.add_argument('--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('--output', type=str, default='.', help='path of output details .csv file')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

print('Init Model ...')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(args.model)['state_dict'])

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
trans = transforms.Compose([transforms.ToTensor(), normalize])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
model.to(device)


def test(testset, batch_size, model, topk, output_path):
    """
    :param testset:         测试数据集
    :param batch_size:      Dataloader 打包的小 batch 大小
    :param model:           网络模型
    :param topk:            每次选取的 top-k 实例个数
    :param output_path:     保存模型文件的目录
    """

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # open output file
    fconv = open(os.path.join(output_path,'pred.csv'), 'w')
    fconv.write('patch_size={},interval={}\n'.format(testset.size, testset.interval))
    fconv.write('grid,prob\n')
    fconv.close()
    # 热图中各个 patch 的信息保存在output_path/pred.csv

    print('Start testing ...')

    # 同训练第一部分
    model.eval()
    probs = torch.FloatTensor(len(test_loader.dataset))
    with torch.no_grad():
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, input in bar:
            bar.set_postfix(batch="[{}/{}]".format(i + 1, len(test_loader)))
            output = F.softmax(model(input.to(device)), dim=1)
            probs[i * batch_size:i * batch_size + input.size(0)] = output.detach()[:, 1].clone()

    probs = probs.cpu().numpy()
    groups = np.array(testset.imageIDX)
    patches = np.array(testset.patches)

    order = np.lexsort((probs, groups))
    groups = groups[order]
    probs = probs[order]
    patches = patches[order]

    index = np.empty(len(groups), 'bool')
    index[-topk:] = True
    index[:-topk] = groups[topk:] != groups[:-topk]

    max_probs = probs[index]
    max_patches = patches[index]

    # 生成热图
    for i, img in enumerate(testset.images):
        mask = np.zeros((img.shape[0], img.shape[1]))
        for idx in range(topk):
            patch_mask = np.full((testset.size, testset.size), max_probs[idx + i * topk])
            grid = (int(max_patches[idx + i * topk][0]), int(max_patches[idx + i * topk][1]))
            mask[grid[0] : grid[0] + testset.size,
                 grid[1] : grid[1] + testset.size] = patch_mask
            # 输出信息
            print("prob_{}:{}".format(i, max_probs[idx + i * topk]))
            fconv = open(os.path.join(args.output, 'pred.csv'), 'a')
            fconv.write('{},{}\n'.format(grid, max_probs[idx + i * topk]))
            fconv.close()

        mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
        img = img * 0.5 + mask * 0.5
        Image.fromarray(np.uint8(img)).save('output/test_{}.png'.format(i))


if __name__ == "__main__":
    from datasets import LystoTestset

    print('Loading test set ...')
    imageSet_test = LystoTestset(filepath="LYSTO/testing.h5", transform=trans,
                            interval=args.interval, size=args.patch_size, num_of_imgs=5)

    test(imageSet_test, batch_size=args.batch_size, model=model, topk=args.topk, output_path='.')