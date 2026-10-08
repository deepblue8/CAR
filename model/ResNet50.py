import torch.nn as nn
import torchvision.models as models
import torch


class MyResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(MyResNet50, self).__init__()
        self.ResNet = models.resnet50(pretrained=pretrained)

    def forward(self, x):
        with torch.no_grad():
            x = self.ResNet.conv1(x)
            x = self.ResNet.bn1(x)
            x = self.ResNet.relu(x)
            x = self.ResNet.maxpool(x)

            x = self.ResNet.layer1(x)
            x = self.ResNet.layer2(x)
            x = self.ResNet.layer3(x)
            x = self.ResNet.layer4(x)
        return x
