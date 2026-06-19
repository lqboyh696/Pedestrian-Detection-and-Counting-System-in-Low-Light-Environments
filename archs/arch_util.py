# archs/arch_util.py - 模型构建工具模块
# 提供自定义的 LayerNorm2d 层和 CustomSequential 容器

import torch
import numpy as np
from torch import nn as nn
from torch.nn import init as init

# 自定义 LayerNorm 的前向/反向传播实现
# 与标准 BN 不同：LayerNorm 在通道维度上归一化，更适合小 batch 场景
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        # 在通道维度计算均值和方差
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        # 应用可学习的缩放和偏移
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

# 2D 版 Layer Normalization：在 (C, H, W) 维度上归一化
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# 自定义顺序容器：支持 adapter 参数传递
class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, use_adapter=False):
        for module in self.modules_list:
            # 如果模块支持 adapter，则设置 adapter 开关
            if hasattr(module, 'set_use_adapters'):
                module.set_use_adapters(use_adapter)
            x = module(x)
        return x

if __name__ == '__main__':
    
    pass