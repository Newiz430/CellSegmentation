from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dataset import categorize, de_categorize

def inference_tiles(loader, model, device, epoch=None, total_epochs=None, mode='train'):
    """前馈推导一次模型，获取实例分类概率。"""

    model.setmode("tile")
    model.eval()

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        tile_bar = tqdm(loader, desc="tile forwarding")
        if mode == 'test':
            tile_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
        for i, input in enumerate(tile_bar):
            if mode == 'train':
                input = input[0]
            # softmax 输出 [[a,b],[c,d]] shape = batch_size*2
            output = model(input.to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] 取出 softmax 得到的概率，产生：[b, d, ...]
            # input.size(0) 返回 batch 中的实例数量
            probs[i * loader.batch_size:i * loader.batch_size + input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def sample(trainset, probs, tiles_per_pos, topk_neg):
    """找出概率为 top-k 的图像块，制作迭代使用的数据集。"""

    groups = np.array(trainset.tileIDX)
    order = np.lexsort((probs, groups))

    index = np.empty(len(trainset), 'bool')
    for i in range(len(trainset)):
        topk = topk_neg if trainset.labels[groups[i]] == 0 else trainset.labels[groups[i]] * tiles_per_pos
        index[i] = groups[i] != groups[(i + topk) % len(groups)]

    p, n = trainset.make_train_data(list(order[index]))
    print("Training data is sampled. (Pos samples: {} | Neg samples: {})".format(p, n))


def inference_image(loader, model, device, epoch=None, total_epochs=None, mode='train', cls_limit=True):
    """前馈推导一次模型，获取图像级的分类概率和回归预测值。"""

    model.setmode("image")
    model.eval()

    # probs = torch.tensor(())
    # nums = torch.tensor(())
    categories = np.array(())
    counts = np.array(())
    with torch.no_grad():
        image_bar = tqdm(loader, desc="image forwarding")
        if epoch is not None and total_epochs is not None:
            image_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
        for i, data in enumerate(image_bar):
            if mode == 'train':
                data = data[0]
            output = model(data.to(device))
            output_cls = F.softmax(output[0], dim=1)
            output_cls = output_cls.detach().clone().cpu()
            output_reg = output[1].detach()[:, 0].clone().cpu()

            # probs = torch.cat((probs, output_cls), dim=0)  # probs: [len(dataset), 7]
            # nums = torch.cat((nums, output_reg), dim=0)  # nums: [len(dataset)]

            output_cls = np.round(output_cls.numpy()).astype(int)
            output_reg = np.round(output_reg.numpy()).astype(int)
            cat_labels = np.argmax(output_cls, axis=1)

            if cls_limit:
                for i, x in enumerate(output_reg):
                   if categorize(x) > cat_labels[i]:
                       output_reg[i] = de_categorize(cat_labels[i])[1]
                   elif categorize(x) < cat_labels[i]:
                       output_reg[i] = de_categorize(cat_labels[i])[0]

            categories = np.concatenate((categories, cat_labels))
            counts = np.concatenate((counts, output_reg))

    # return probs.numpy(), nums.numpy()
    return categories, counts
