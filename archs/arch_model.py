# archs/arch_model.py - DarkIR 模型的子模块定义
# 包含编码器块(EBlock)、解码器块(DBlock)、频域MLP等核心组件

import torch
import torch.nn as nn

try:
    from .arch_util import LayerNorm2d
except:
    from arch_util import LayerNorm2d

# 简单门控机制：将通道分成两半，逐元素相乘
# 用于模拟信息门控，增强特征表达
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# 频域多层感知机：在傅里叶频域中对幅度谱进行卷积处理
# 利用频域信息捕获全局特征，增强去噪和增强效果
class FreMLP(nn.Module):
    
    def __init__(self, nc, expand = 2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        # 实数 FFT 转换到频域
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)      # 幅度谱
        pha = torch.angle(x_freq)    # 相位谱
        # 仅在幅度谱上做卷积处理，保持相位不变
        mag = self.process1(mag)
        # 重构复数频谱并逆傅里叶变换回空间域
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

# 分支模块：带空洞卷积的单路分支
# 支持不同 dilation 率，捕获多尺度感受野
class Branch(nn.Module):

    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # 空洞深度可分离卷积
        )
    def forward(self, input):
        return self.branch(input)
    
# 解码器块 (DBlock)：带多尺度空洞卷积分支 + 通道注意力 + 残差连接
class DBlock(nn.Module):
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        # 定义两个并行分支
        self.dw_channel = DW_Expand * c 

        # 1x1 卷积扩展通道
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # 可选的额外深度可分离卷积
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity()
        # 多尺度空洞卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        # 通道注意力：自适应平均池化 + 1x1 卷积
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()  # 第一个门控
        self.sg2 = SimpleGate()  # 第二个门控（FFN 部分）
        # 1x1 卷积压缩回原通道数
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        # FFN 前馈网络：扩展 + 压缩
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # 可学习的残差缩放参数
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        
    def forward(self, inp, adapter = None):

        y = inp
        x = self.norm1(inp)
        # 第一分支：多尺度空洞卷积
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)  # 多分支结果求和
        
        z = self.sg1(z)  # 门控激活
        x = self.sca(z) * z  # 通道注意力加权
        x = self.conv3(x)  # 压缩回原通道
        y = inp + self.beta * x  # 残差连接
        # 第二分支：FFN 前馈网络
        x = self.conv4(self.norm2(y))
        x = self.sg2(x)  # 门控激活
        x = self.conv5(x)  # 压缩回原通道
        x = y + x * self.gamma  # 残差连接
        
        return x 

class EBlock(nn.Module):
    '''
    编码器块：结构与 DBlock 类似，但 FFN 部分使用频域 MLP
    '''
    
    def __init__(self, c, DW_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        # 定义分支参数
        self.dw_channel = DW_Expand * c 
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
                
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        # 第一分支：多尺度空洞卷积
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)  # 多分支空洞卷积求和
        
        z = self.sg1(z)      # 门控激活
        x = self.sca(z) * z  # 通道注意力加权
        x = self.conv3(x)    # 压缩回原通道
        y = inp + self.beta * x  # 残差连接
        # 第二分支：频域 MLP 代替普通 FFN
        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)  # 频域处理
        x = y * x_freq 
        x = y + x * self.gamma  # 残差连接

        return x 

if __name__ == '__main__':
    
    # 测试 EBlock 模块的计算复杂度
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net  = EBlock(c = img_channel,
                            dilations = dilations,
                            extra_depth_wise=extra_depth_wise)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    output = net(torch.randn((4, 3, 256, 256)))
    print(macs, params)

    channels = 128
    resol = 32
    ksize = 5