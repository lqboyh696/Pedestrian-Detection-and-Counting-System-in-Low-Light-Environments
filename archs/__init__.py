# archs/__init__.py - 模型架构包入口
# 提供统一的模型创建工厂函数

from ptflops import get_model_complexity_info

from .DarkIR import DarkIR   

def create_model(opt, rank, adapter = False, use_ddp=False):
    '''根据配置字典创建 DarkIR 模型实例，并计算计算复杂度和参数量'''
    name = opt['name']

    # 按配置参数实例化 DarkIR 网络
    model = DarkIR(img_channel=opt['img_channels'], 
                    width=opt['width'], 
                    middle_blk_num_enc=opt['middle_blk_num_enc'],
                    middle_blk_num_dec=opt['middle_blk_num_dec'], 
                    enc_blk_nums=opt['enc_blk_nums'],
                    dec_blk_nums=opt['dec_blk_nums'], 
                    dilations=opt['dilations'],
                    extra_depth_wise=opt['extra_depth_wise'])

    # 主进程（rank=0）打印模型信息
    if rank ==0:
        print(f'Using {name} network')

        input_size = (3, 256, 256)
        macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
        print(f'Computational complexity at {input_size}: {macs}')
        print('Number of parameters: ', params)    
    else:
        macs, params = None, None

    # 将模型移动到指定设备
    model.to(rank)

    return model, macs, params

__all__ = ['create_model']
