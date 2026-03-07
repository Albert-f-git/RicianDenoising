import torch
import torch.nn as nn

class RicianNet(nn.Module):
    def __init__(self):
        super(RicianNet, self).__init__()
        
        # ==========================================
        # Sub-RicianNet 1: 无 BN，完全拟合非线性分布
        # ==========================================
        self.sub1_cr = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3), nn.ReLU(inplace=True)
        )
        self.sub1_c5 = nn.Conv2d(128, 1, kernel_size=7, padding=3)

        # ==========================================
        # Sub-RicianNet 2: 引入 BN，处理稀疏特征
        # ==========================================
        self.sub2_c6 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        
        # ResBNet 模块 (4个 CBR + 1个 CB)
        self.sub2_resb_cbr = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, padding=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        
        self.sub2_cb11 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64)
        )
        
        # 结尾映射
        self.sub2_c12_13 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=7, padding=3)
        )

    def forward(self, x):
        # 第一子网络前向传播 (带残差连接)
        out1 = self.sub1_cr(x)
        out1 = self.sub1_c5(out1)
        sub1_out = x + out1

        # 第二子网络前向传播
        c6_out = self.sub2_c6(sub1_out)
        
        resb_out = self.sub2_resb_cbr(c6_out)
        cb11_out = self.sub2_cb11(resb_out)
        
        # ResBNet 的残差连接
        sub2_res_out = c6_out + cb11_out
        
        # 最终输出
        final_out = self.sub2_c12_13(sub2_res_out)
        return final_out

if __name__ == "__main__":
    model = RicianNet()
    print("模型初始化成功，参数量检查...")
    dummy_input = torch.randn(2, 1, 41, 41)
    print("输出形状:", model(dummy_input).shape) # 预期: [2, 1, 41, 41]