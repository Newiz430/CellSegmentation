## Multiple Instance Learning for Immune Cell Image Segmentation with Counting Labels

<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.7.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>

This is the new multi-stage auto-immunoquantitative analytical model we proposed. 

Special thanks to Dr. Cheng [@ShenghuaCheng](https://github.com/ShenghuaCheng) for contributing to this work 
and [WNLO](http://wnlo.hust.edu.cn/) for platform provision. 

### New MIL for immune cell pathological images

Taking immunohistochemistry-stained digital cell images as input, the model is merely supervised by positive cell counting labels and transforms whole-image (bag) level counting results into superpixel (instance) level classification results via the specifically designed adaptive top-k instance selection strategy.

### Network frame

- Stage 1: Image-wise regressive positive cell counter
- Stage 2: Superpixel-wise tile instance classifier
- Stage 3: Pixel-wise segmentation encoder-decoder network

![](figures/network_frame.png)

### Adaptive top-k selection

Supervised by counting labels. Needs a little elementary geometry. 

![](figures/topk.png)

### Mask refinement

Instance classifier provides us semantic information of positive cells. 
HSV channel separation and thresholding provide us fine-grained profile of positive cells.

### Grand Challenge results

Kappa = 0.9319, 4th in **Lymphocyte Assessment Hackathon** (LYSTO) Challenge. [Leaderboard](https://lysto.grand-challenge.org/evaluation/challenge/leaderboard/)

We also tested our localization method in [LYON19](https://lyon19.grand-challenge.org/). 
(I wonder what this acronym stands for)

### Dataset

Visit [LYSTO](https://lysto.grand-challenge.org/) to get data.

DLBCL-IHC is not available for now. I won't upload this until this work is accepted. 

### Quick Start

Well this project is not well refactored for now and if you really wanna try this...

- Add image data in `./data`
- Train cell counter by `python train_image.py`
- Test your counter by `python test_count.py`
- Train tile classifier by `python train_tile.py`
- Test the classifier and get heatmaps by `python test_tile.py`
- Train segmentation network by `python train_seg.py`
- Test segmentation network and get masks by `python test_seg.py --draw_masks`
- Test segmentation network and get localization points by `python test_seg.py --detect`

and use arguments you like. You can find arguments list in the source code file. 

### Citing

... under construction ... stay tuned. 

> 2021-2022 By Newiz
