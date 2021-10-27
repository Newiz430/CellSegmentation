from torchsummary import summary
from model.resnet import MILresnet18

if __name__ == "__main__":
    model = MILresnet18(pretrained=False)
    model.setmode("patch")
    print("patch mode:\n")
    summary(model, input_size=(3, 32, 32), batch_size=32)
    model.setmode("slide")
    print("slide mode:\n")
    summary(model, input_size=(3, 299, 299), batch_size=32)