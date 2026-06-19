# inference_DarkIR.py - DarkIR 低光照增强模型推理模块
# 提供图片/视频的批量增强、单张/单帧增强等功能

import os
import numpy as np
import cv2 as cv
from PIL import Image
from options.options import parse
import torch.nn.functional as F
from torchvision import transforms
import torch
from tqdm import tqdm
from torchvision.transforms import Resize
from archs import create_model
from ptflops import get_model_complexity_info

# 设备自动选择：CUDA > MPS（Apple GPU） > CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

# --- 数据格式转换工具 ---

def path_to_tensor(path):
    '''从文件路径加载图像并转为 [1, 3, H, W] 张量'''
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    return img

def array_to_tensor(frame):
    '''OpenCV BGR numpy 数组转为 [1, 3, H, W] 张量'''
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    return tensor_frame

def tensor_to_array(tensor):
    '''[1, 3, H, W] 张量转为 OpenCV BGR numpy 数组'''
    array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frame = (array * 255).astype(np.uint8)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    return frame

def normalize_tensor(tensor):
    '''Min-Max 归一化到 [0, 1] 范围'''
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value) / (max_value)
    return output

def save_tensor(tensor, path):
    '''保存张量为图像文件'''
    tensor = tensor.squeeze(0)
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple=8):
    '''右侧/底部零填充，使 H 和 W 是 multiple 的整数倍'''
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

# --- 模型加载与管理 ---

def load_model(model, path_weights):
    '''加载 DarkIR 模型权重（支持 DDP 包装的权重）'''
    map_location = device
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    weights = checkpoints['params']
    
    # 移除 DataParallel 的 'module.' 前缀
    if any(key.startswith('module.') for key in weights.keys()):
        weights = {key.replace('module.', ''): value for key, value in weights.items()}
    
    macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
    
    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        print(f'Error loading weights: {e}')
        print('Trying to load with strict=False...')
        model.load_state_dict(weights, strict=False)
    
    return model

def init_model(config_path='./options/DarkIR.yml'):
    '''初始化 DarkIR 模型：加载配置、创建模型、加载权重'''
    opt = parse(config_path)
    model, _, _ = create_model(opt['network'], rank=0, use_ddp=False)
    model = load_model(model, path_weights=opt['save']['path'])
    model.eval()
    resize = opt.get('Resize', False)
    return model, resize

def enhance_image(model, tensor, resize=False):
    '''对单张图像张量进行 DarkIR 增强（支持大图先缩小后放大策略）'''
    _, _, H, W = tensor.shape
    
    # 大图先缩小一半处理，再放大回来，节省显存
    if resize and (H >= 1500 or W >= 1500):
        new_size = [int(dim // 2) for dim in (H, W)]
        downsample = Resize(new_size)
    else:
        downsample = torch.nn.Identity()
    
    tensor = downsample(tensor)
    tensor = pad_tensor(tensor)
    
    with torch.no_grad():
        output = model(tensor, side_loss=False)
    
    if resize:
        upsample = Resize((H, W))
    else:
        upsample = torch.nn.Identity()
    
    output = upsample(output)
    output = torch.clamp(output, 0., 1.)
    output = output[:, :, :H, :W]  # 裁剪填充部分
    
    return output

def resize_to_1080p(tensor):
    '''将张量缩放到短边不超过 1080 像素，保证处理速度'''
    _, _, H, W = tensor.shape
    shorter = min(H, W)
    if shorter <= 1080:
        return tensor
    scale = 1080.0 / shorter
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))
    tensor = Resize((new_H, new_W))(tensor)
    print(f'[缩小] {W}x{H} -> {new_W}x{new_H} (短边限制1080)')
    return tensor


def calc_1080p_dims(H, W):
    '''计算短边限制 1080 像素下的目标尺寸'''
    shorter = min(H, W)
    if shorter <= 1080:
        return H, W
    scale = 1080.0 / shorter
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))
    return new_H, new_W

# --- 高层接口：单张/批量/视频增强 ---

def enhance_single_image(model, image_path, output_path, resize=False):
    '''增强单张图像并保存'''
    tensor = path_to_tensor(image_path).to(device)
    tensor = resize_to_1080p(tensor)
    output = enhance_image(model, tensor, resize)
    save_tensor(output, output_path)
    return output_path

def enhance_folder(model, input_folder, output_folder, resize=False):
    '''批量增强文件夹中的所有图像'''
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    path_images = [
        os.path.join(input_folder, path) 
        for path in os.listdir(input_folder) 
        if path.endswith(('.png', '.PNG', '.jpg', '.JPEG', '.JPG'))
    ]
    path_images = [file for file in path_images if not file.endswith('.csv') and not file.endswith('.txt')]
    
    print(f'Found {len(path_images)} images to process')
    
    pbar = tqdm(total=len(path_images))
    
    for path_img in path_images:
        output_path = os.path.join(output_folder, os.path.basename(path_img))
        enhance_single_image(model, path_img, output_path, resize)
        pbar.update(1)
    
    pbar.close()
    print('Finished image enhancement!')

def enhance_video(model, video_path, output_path, resize=False):
    '''增强单个视频文件（>24fps 自动抽帧至 24fps，>1080p 自动缩小）'''
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv.CAP_PROP_FPS)
    
    if original_fps <= 0:
        original_fps = 24.0
    
    # 高帧率视频抽帧至 24fps
    if original_fps > 24:
        output_fps = 24
        sample_interval = original_fps / 24.0
        print(f'[抽帧] {os.path.basename(video_path)}: {original_fps:.1f}fps -> {output_fps}fps')
    else:
        output_fps = int(original_fps)
        sample_interval = 1.0
    
    # 超出 1080p 的视频自动缩小
    out_height, out_width = calc_1080p_dims(frame_height, frame_width)
    need_resize = (out_width != frame_width) or (out_height != frame_height)
    if need_resize:
        print(f'[缩小] {os.path.basename(video_path)}: {frame_width}x{frame_height} -> {out_width}x{out_height} (短边限制1080)')
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, output_fps, (out_width, out_height))
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
    
    next_frame_pos = 0.0
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 抽帧逻辑：按 sample_interval 间隔处理
        if frame_idx >= int(next_frame_pos):
            if need_resize:
                frame = cv.resize(frame, (out_width, out_height))
            tensor = array_to_tensor(frame).to(device)
            tensor = normalize_tensor(tensor)
            output = enhance_image(model, tensor, resize)
            enhanced_frame = tensor_to_array(output)
            out.write(enhanced_frame)
            next_frame_pos += sample_interval
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    out.release()
    pbar.close()
    
    return output_path

def enhance_videos_in_folder(model, input_folder, output_folder, resize=False):
    '''批量增强文件夹中的所有视频'''
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if os.path.splitext(f)[1].lower() in video_extensions
    ]
    
    if not video_files:
        raise ValueError(f"No video files found in {input_folder}")
    
    print(f"Found {len(video_files)} video(s) to process")
    
    for video_path in video_files:
        output_path = os.path.join(output_folder, os.path.basename(video_path))
        enhance_video(model, video_path, output_path, resize)
        print(f'Finished: {os.path.basename(video_path)}')
    
    print('\nAll videos processed successfully!')

# --- 命令行入口 ---

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Script for prediction")
    parser.add_argument('-p', '--config', type=str, default='./options/DarkIR.yml', help='Config file of prediction')
    parser.add_argument('-i', '--inp_path', type=str, default='./inputs', help="Folder path")
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video'], help="Enhancement mode")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        print(f'GPU: Apple MPS (Metal Performance Shaders)')
    
    model, resize = init_model(args.config)
    PATH_RESULTS = './results'
    
    if args.mode == 'image':
        if os.path.isfile(args.inp_path):
            output_path = os.path.join(PATH_RESULTS, os.path.basename(args.inp_path))
            enhance_single_image(model, args.inp_path, output_path, resize)
        else:
            enhance_folder(model, args.inp_path, PATH_RESULTS, resize)
    elif args.mode == 'video':
        if os.path.isfile(args.inp_path):
            output_path = os.path.join(PATH_RESULTS, os.path.basename(args.inp_path))
            enhance_video(model, args.inp_path, output_path, resize)
        else:
            enhance_videos_in_folder(model, args.inp_path, PATH_RESULTS, resize)
