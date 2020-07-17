import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import resnet18
from models.resnet import resnet50
from models.fpem import FPEM_FFM


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.enhance_fusion = FPEM_FFM(backbone_out_channels=[64, 128, 256, 512])

    def forward(self, input):
        _, _, h, w = input.size()
        output = self.backbone(input)
        output = self.enhance_fusion(output)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        return output



if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    # model_config = {
    #     'backbone': 'shufflenetv2',
    #     'fpem_repeat': 4,  # fpem模块重复的次数
    #     'pretrained': True,  # backbone 是否使用imagesnet的预训练模型
    #     'result_num': 7,
    #     'segmentation_head': 'FPEM_FFM'  # 分割头，FPN or FPEM_FFM
    # }
    model = Model().to(device)
    y = model(x)
    print(y.shape)
    # print(model)
    # torch.save(model.state_dict(), 'PAN.pth')