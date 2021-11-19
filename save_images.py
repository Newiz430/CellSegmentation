import os
import numpy as np
from PIL import Image


def save_images(dataset, prefix, output_path, num_of_imgs=0):
    """
    把 hdf5 数据中的图像以 <name>_<idx>.png 的名称导出。
    :param dataset:     LystoDataset
    :param prefix:      所有图片共享的名称部分
    :param output_path: 输出图片路径
    :param num_of_imgs: 选择输出图片的数目（前 n 个）
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i, img in enumerate(dataset.images):
        if num_of_imgs != 0 and i == num_of_imgs:
            break
        name = '{}_{}_{}cells.png'.format(prefix, i, dataset.labels[i]) if hasattr(dataset, 'labels') \
            else '{}_{}.png'.format(prefix, i)
        Image.fromarray(np.uint8(img)).save(os.path.join(output_path, name))


if __name__ == "__main__":
    from dataset.dataset import LystoDataset, LystoTestset

    # imageSet_test = LystoTestset(filepath="data/test.h5")
    # save_images(imageSet_test, 'test', './data/test')

    set = LystoDataset(filepath="data/training.h5", train=False)
    save_images(set, 'val', './data/val')
