# model.py - Zero-DCE 图像增强网络模型定义
# 实现 Zero-DCE 低光照增强网络：使用深度可分离卷积和迭代增强曲线

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 深度可分离卷积模块：depthwise + pointwise 组合，大幅减少参数量
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        # depthwise 卷积：每个输入通道单独卷积
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        # pointwise 卷积：1x1 卷积实现跨通道信息融合
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

# Zero-DCE 增强网络主体：无池化层的轻量级 U-Net 结构
# 通过预测像素级增强曲线参数 x_r 来实现低光照图像增强
class enhance_net_nopool(nn.Module):
	def __init__(self,scale_factor):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		self.scale_factor = scale_factor  # 下采样因子，用于加速处理
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		number_f = 32  # 基础特征通道数

		# 7 个深度可分离卷积层，构建编解码结构
		self.e_conv1 = CSDN_Tem(3,number_f) 
		self.e_conv2 = CSDN_Tem(number_f,number_f) 
		self.e_conv3 = CSDN_Tem(number_f,number_f) 
		self.e_conv4 = CSDN_Tem(number_f,number_f) 
		self.e_conv5 = CSDN_Tem(number_f*2,number_f)   # 跳跃连接：拼接 e_conv3 + e_conv4
		self.e_conv6 = CSDN_Tem(number_f*2,number_f)   # 跳跃连接：拼接 e_conv2 + e_conv5
		self.e_conv7 = CSDN_Tem(number_f*2,3)          # 跳跃连接：拼接 e_conv1 + e_conv6，输出 3 通道曲线参数

	# 迭代增强函数：通过 8 次迭代的高阶曲线逐步提亮图像
	def enhance(self, x,x_r):

		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
		x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + x_r*(torch.pow(x,2)-x)	
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image = x + x_r*(torch.pow(x,2)-x)	

		return enhance_image
		
	def forward(self, x):
		# 下采样阶段：按 scale_factor 缩小输入以加速推理
		if self.scale_factor==1:
			x_down = x
		else:
			x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')

		# 编码器：7 层深度可分离卷积 + 跳跃连接
		x1 = self.relu(self.e_conv1(x_down))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		# 输出增强曲线参数 x_r，范围 [-1, 1]
		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		# 上采样恢复原始分辨率
		if self.scale_factor==1:
			x_r = x_r
		else:
			x_r = self.upsample(x_r)
		# 应用迭代增强曲线
		enhance_image = self.enhance(x,x_r)
		return enhance_image,x_r
