import torch
import torch.nn as nn
import torch.nn.functional as F
from model.video_swin_transformer import SwinTransformer3D


class VSTModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SwinTransformer3D()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.backbone(x)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = self.fc2(output)
        # output = self.sigmoid(output)
        return output



if __name__ == '__main__':
    model = VSTModel(num_classes=2)
    dummy_x = torch.rand(1, 3, 32, 224, 224)
    logits = model(dummy_x)
    print(logits.shape)


