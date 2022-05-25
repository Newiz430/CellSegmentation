from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dataset import categorize, de_categorize


def inference_tiles(loader, model, device, epoch=None, total_epochs=None, mode='train'):
    """Forward inference to obtain instance classification probs. """

    model.eval()

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        tile_bar = tqdm(loader, desc="tile forwarding")
        if epoch is not None:
            tile_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
        for i, input in enumerate(tile_bar):
            if mode == 'train':
                input = input[0]
            # softmax -- [[a,b],[c,d]] shape = batch_size * 2
            output = model(input.to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] extracts probs given by softmax: [b, d, ...]
            # input.size(0) returns the number of instances in one batch
            probs[i * loader.batch_size:i * loader.batch_size + input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def sample(trainset, probs, tiles_per_pos, topk_neg, pos_neg_ratio):
    """Select top-k superpixels to create a instance training set. """

    groups = np.array(trainset.tileIDX)
    order = np.lexsort((probs, groups))

    index = np.empty(len(trainset), 'bool')
    for i in range(len(trainset)):
        topk = topk_neg if trainset.labels[groups[i]] == 0 else trainset.labels[groups[i]] * tiles_per_pos
        index[i] = groups[i] != groups[(i + topk) % len(groups)]

    p, n = trainset.make_train_data(list(order[index]), pos_neg_ratio)
    print("Training data is sampled. (Pos samples: {} | Neg samples: {})".format(p, n))


def inference_image(loader, model, device, epoch=None, total_epochs=None, mode='train', cls_limit=False,
                    return_id=False):
    """Forward inference to obtain image-level classification probs and pos cell counts. """

    model.eval()

    # probs = torch.tensor(())
    # nums = torch.tensor(())
    ids = np.array(())
    categories = np.array(())
    counts = np.array(())
    with torch.no_grad():
        image_bar = tqdm(loader, desc="image forwarding")
        if epoch is not None and total_epochs is not None:
            image_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
        for i, data in enumerate(image_bar):
            if mode == 'train':
                data = data[0]
            else:
                batch_ids, data = data
                ids = np.concatenate((ids, batch_ids))
            output = model(data.to(device))
            output_cls = F.softmax(output[0], dim=1)
            output_cls = output_cls.detach().clone().cpu()
            output_reg = output[1].detach()[:, 0].clone().cpu()

            # probs = torch.cat((probs, output_cls), dim=0)  # probs: [len(dataset), 7]
            # nums = torch.cat((nums, output_reg), dim=0)  # nums: [len(dataset)]

            output_reg = np.round(output_reg.numpy()).astype(int)
            cat_labels = np.argmax(output_cls, axis=1)

            if cls_limit:
                for i, x in enumerate(output_reg):
                    # if cat_labels[i] == 0:
                    #     output_reg[i] = 0  # use cls branch for artifact images
                    # if cat_labels[i] > 4:
                    if categorize(x) > cat_labels[i]:
                        output_reg[i] = de_categorize(cat_labels[i])[1]
                    elif categorize(x) < cat_labels[i]:
                        output_reg[i] = de_categorize(cat_labels[i])[0]

            categories = np.concatenate((categories, cat_labels))
            counts = np.concatenate((counts, output_reg))

    if return_id:
        return ids, categories, counts
    else:
        # return probs.numpy(), nums.numpy()
        return categories, counts


def inference_image_cls(loader, model, device, epoch=None, total_epochs=None, mode='train'):

    model.eval()

    categories = np.array(())
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
            cat_labels = np.argmax(output_cls, axis=1)

            categories = np.concatenate((categories, cat_labels))

    return categories  # [n, 1]


def inference_image_reg(loader, model, device, epoch=None, total_epochs=None, mode='train'):

    model.eval()

    nums = torch.tensor(())
    with torch.no_grad():
        image_bar = tqdm(loader, desc="image forwarding")
        if epoch is not None and total_epochs is not None:
            image_bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs))
        for i, data in enumerate(image_bar):
            if mode == 'train':
                data = data[0]
            output = model(data.to(device))
            output_reg = output[1].detach()[:, 0].clone().cpu()

            nums = torch.cat((nums, output_reg), dim=0)  # nums: [len(dataset)]

    return nums.numpy()


def inference_seg(loader, model, device, mode='train'):

    model.eval()

    masks = []
    with torch.no_grad():
        seg_bar = tqdm(loader, desc="image segmenting")
        for i, data in enumerate(seg_bar):
            output = model(data.to(device))
            if mode == 'test':
                output = F.softmax(output, dim=1)[:, 1]  # note: channel 1 for pos_mask=1 and bg=0
            masks.append(output.cpu().numpy())

    return np.concatenate(masks)
