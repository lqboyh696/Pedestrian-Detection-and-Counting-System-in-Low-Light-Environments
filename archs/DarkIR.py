# archs/DarkIR.py - DarkIR 低光照图像增强网络主体
# 基于 U-Net 架构：编码器(EBlock) → 中间层 → 解码器(DBlock)
# 使用 PixelShuffle 上采样，支持跳跃连接和 side loss

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from arch_model import EBlock, DBlock
    from arch_util import CustomSequential
except:
    from archs.arch_model import EBlock, DBlock
    from .arch_util import CustomSequential

class DarkIR(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=32,                        # 基础特征通道数
                 middle_blk_num_enc=2,            # 编码器中间块数量
                 middle_blk_num_dec=2,            # 解码器中间块数量
                 enc_blk_nums=[1, 2, 3],          # 各层编码器块数量
                 dec_blk_nums=[3, 1, 1],          # 各层解码器块数量
                 dilations = [1, 4, 9],           # 空洞卷积膨胀率
                 extra_depth_wise = True):         # 是否使用额外深度可分离卷积
        super(DarkIR, self).__init__()
        
        # 输入/输出卷积层
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        # 编码器下采样路径
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            # 2倍下采样 + 2倍通道扩展
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # 中间瓶颈层
        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        # 解码器上采样路径
        for num in dec_blk_nums:
            # PixelShuffle 上采样：通道减半，分辨率翻倍
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        # 计算需要的填充大小，保证下采样层数后尺寸能被整除
        self.padder_size = 2 ** len(self.encoders)        
        
        # side loss 输出层：用于计算中间监督损失
        self.side_out = nn.Conv2d(in_channels = width * 2**len(self.encoders), out_channels = img_channel, 
                                kernel_size = 3, stride=1, padding=1)
        
    def forward(self, input, side_loss = False, use_adapter = None):

        _, _, H, W = input.shape

        # 确保输入尺寸是 padder_size 的整数倍
        input = self.check_image_size(input)
        x = self.intro(input)
        
        # 编码器路径：逐层下采样并保存跳跃连接
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # 应用中间编码器变换
        x_light = self.middle_blks_enc(x)
        
        if side_loss:
            out_side = self.side_out(x_light)
        # 应用解码器变换
        x = self.middle_blks_dec(x_light)
        x = x + x_light  # 残差连接

        # 解码器路径：逐层上采样并拼接跳跃连接
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip  # 跳跃连接
            x = decoder(x)

        x = self.ending(x)
        x = x + input  # 全局残差连接
        out = x[:, :, :H, :W] # 恢复原始图像尺寸
        if side_loss:
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        '''通过反射填充使 H, W 能被 padder_size 整除'''
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x      

if __name__ == '__main__':
    
    # 测试 DarkIR 模型的计算复杂度
    img_channel = 3
    width = 32
    
    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net = DarkIR(img_channel=img_channel, 
                  width=width, 
                  middle_blk_num_enc=middle_blk_num_enc,
                  middle_blk_num_dec= middle_blk_num_dec,
                  enc_blk_nums=enc_blks, 
                  dec_blk_nums=dec_blks,
                  dilations = dilations,
                  extra_depth_wise = extra_depth_wise)
    
    new_state_dict = net.state_dict()

    inp_shape = (3, 256, 256)

    net.load_state_dict(new_state_dict)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(macs, params)    
    
    weights = net.state_dict()
    adapter_weights = {k: v for k, v in weights.items() if 'adapter' not in k}
