import os
import numpy as np
from PIL import Image

def save_images(dataset, name, output_path='.', num_of_imgs=0):
    """
    把 hdf5 数据中的图像以 <name>_<idx>.png 的名称导出。
    :param dataset:     LystoDataset
    :param name:        所有图片共享的名称部分
    :param output_path: 输出图片路径
    :param num_of_imgs: 选择输出图片的数目（前 n 个）
    """

    for i, img in enumerate(dataset.images):
        if num_of_imgs != 0 and i == num_of_imgs:
            break
        Image.fromarray(np.uint8(img)).save(os.path.join(output_path, '{}_{}.png'.format(name, i)))


if __name__ == "__main__":
    from dataset.datasets import LystoTestset

    imageSet_test = LystoTestset(filepath="data/testing.h5", interval=20, size=32, num_of_imgs=20)
    save_images(imageSet_test, 'test', './image', 20)