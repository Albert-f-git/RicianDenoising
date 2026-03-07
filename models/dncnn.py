import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # 第一层：Conv + ReLU (注意：第一层不需要 Batch Normalization)
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层：Conv + BN + ReLU (重复 15 次)
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, 
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        # 最后一层：Conv (仅输出预测的噪声残差，无激活函数)
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, 
                                kernel_size=kernel_size, padding=padding, bias=False))
        
        # 使用 nn.Sequential 将所有层串联
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # 核心数学逻辑：
        # 网络 self.dncnn(x) 预测的是“噪声”
        # 干净图像 = 带噪图像 - 预测的噪声
        residual = self.dncnn(x)
        return x - residual