# options/options.py - YAML 配置文件解析模块
# 为 DarkIR 等模型提供 YAML 配置文件的有序字典解析能力

import os
import yaml
from collections import OrderedDict

# 优先使用 C 语言版本的 YAML 加载器以提升速度
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''让 YAML 解析支持 Python 的有序字典，保持配置文件中键的顺序'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

# 注册有序字典支持
Loader, Dumper = OrderedYaml()

def parse(opt_path):
    '''
    从 YAML 配置文件解析出有序字典
    '''
    if not os.path.isfile(opt_path): raise ValueError('The config file does not exist!')
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt


# 模块自测试入口
if __name__ == '__main__':
    
    path_yaml = './train/NBDN.yml'
    with open(path_yaml, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    opt = parse(path_yaml)
    print(type(opt['network']['width']))