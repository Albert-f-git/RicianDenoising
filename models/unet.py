import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.channel_attention_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_attention(x)
        max_out = self.channel_attention_max(x)
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""
    # 去除了内部的 Attention 逻辑，让它回归纯粹的特征提取
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], use_attention=True):
        super(UNet, self).__init__()
        self.use_attention = use_attention
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 为跳跃连接准备 Attention 模块列表
        if self.use_attention:
            self.attentions = nn.ModuleList()
            # 因为解码是从深层到浅层，所以这里的特征通道数也要反转匹配
            for feature in reversed(features):
                self.attentions.append(CBAM(feature))

        # 1. 编码器 (纯净版，不包含 Attention)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 3. 解码器
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        identity = x  

        # --- 编码过程 (干净地下采样) ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # 存下原始特征图
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] 

        # --- 解码过程 ---
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            # 只在跳跃连接拼接前，对其应用 CBAM 过滤！
            if self.use_attention:
                skip_connection = self.attentions[i//2](skip_connection)

            # 鲁棒性 Padding
            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_x)

        # --- 全局残差学习 ---
        noise_pred = self.final_conv(x)
        return identity - noise_pred