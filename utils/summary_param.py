import os
import torch
from torch import nn
from torchsummary import summary
from model import encoders

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)


def summary_param(model, batch_size, gpu='0'):

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.setmode("tile")
    print("tile mode:\n")
    summary(model, input_size=(3, 32, 32), batch_size=batch_size)
    model.setmode("image")
    print("\nimage mode:\n")
    summary(model, input_size=(3, 299, 299), batch_size=batch_size)


if __name__ == "__main__":

    model = encoders['MILresnet50']
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    summary_param(model, 32)