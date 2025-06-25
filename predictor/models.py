import torch.nn as nn
import torchvision.models as models

class CustomNeuralNetResNet(nn.Module):
    def __init__(self, outputs_number):
        super(CustomNeuralNetResNet, self).__init__()
        self.net = models.resnet50(weights=None)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, outputs_number)  # Overriding final FC layer

    def forward(self, x):
        return self.net(x)