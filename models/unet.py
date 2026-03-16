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
        # 通道注意力：同时使用AvgPool和MaxPool
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
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if self.use_attention:
            self.attention = CBAM(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], use_attention=True):
        """
        为了防止在普通显卡上爆显存，我们将通道数适度缩减为 32 起步 (原版是 64 起步)。
        这对于单通道的灰度 MRI 去噪已经具备了压倒性的特征提取能力。
        """
        super(UNet, self).__init__()
        self.use_attention = use_attention
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -----------------------------------------
        # 1. 编码器 (Downsampling)
        # -----------------------------------------
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, use_attention=use_attention))
            in_channels = feature

        # -----------------------------------------
        # 2. 瓶颈层 (Bottleneck)
        # -----------------------------------------
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, use_attention=use_attention)

        # -----------------------------------------
        # 3. 解码器 (Upsampling + Skip Connections)
        # -----------------------------------------
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature, use_attention=use_attention))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        identity = x  # 🌟 保存一份原始输入的备份，用于全局残差学习

        # --- 编码过程 ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- 瓶颈过程 ---
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # 反转列表，方便解码器逐层提取

        # --- 解码过程 ---
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            # 🛠️ 鲁棒性设计：如果输入尺寸不是完美的 2 的幂次，这里进行动态 Padding 填补边缘
            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            # 在通道维度 (dim=1) 进行跳跃连接的高维拼接
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_x)

        # --- 全局残差学习 (Global Residual Learning) ---
        noise_pred = self.final_conv(x)
        
        # 让网络预测噪声，然后用原图减去噪声得到干净图像
        return identity - noise_pred