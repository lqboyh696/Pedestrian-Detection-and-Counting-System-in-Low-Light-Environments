'''
# 1. 单张图像增强 (DarkIR)
python main.py --mode image --input inputs/test.jpg --output results/enhanced.jpg
# 2. 批量图片文件夹增强 (DarkIR)
python main.py --mode batch --input inputs/ --output results/
# 3. 单个视频增强 (可选择 DarkIR 或 Zero-DCE)
python main.py --mode video --input videos/test.mp4 --output results/enhanced.mp4 --model darkir
python main.py --mode video --input videos/test.mp4 --output results/enhanced.mp4 --model zero_dce
# 4. 批量视频文件夹增强 (可选择 DarkIR 或 Zero-DCE)
python main.py --mode video_batch --input videos/ --output results/ --model darkir
python main.py --mode video_batch --input videos/ --output results/ --model zero_dce
# 5. 摄像头实时增强 (Zero-DCE)
python main.py --mode camera --camera_id 0
python main.py --mode camera --camera_id 0 --save_video --output camera.mp4
'''

import os
import sys
import argparse

from inference_DarkIR import (
    init_model as init_darkir,
    enhance_single_image as darkir_enhance_single_image,
    enhance_folder as darkir_enhance_folder,
    enhance_video as darkir_enhance_video,
    enhance_videos_in_folder as darkir_enhance_videos_in_folder,
)

from inference_Zero_DCE import create_inference_engine


VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']


def parse_args():
    parser = argparse.ArgumentParser(description='LLE 低光图像增强工具')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['image', 'batch', 'video', 'video_batch', 'camera'],
                        help='运行模式: image(单张图片), batch(批量图片), video(单个视频), video_batch(批量视频), camera(摄像头)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入路径（图像/视频/文件夹），camera模式不需要')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径（可选）')
    parser.add_argument('--model', type=str, default='darkir',
                        choices=['darkir', 'zero_dce'],
                        help='增强模型: darkir(DarkIR) / zero_dce(Zero-DCE)，仅video和video_batch模式有效')
    parser.add_argument('--config', type=str, default='./options/DarkIR.yml',
                        help='DarkIR配置文件路径（仅darkir模型有效，默认: ./options/DarkIR.yml）')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='摄像头ID（仅camera模式有效，默认: 0）')
    parser.add_argument('--save_video', action='store_true',
                        help='是否保存摄像头增强视频（仅camera模式有效）')
    parser.add_argument('--no_display', action='store_true',
                        help='不显示画面（仅camera模式有效）')

    return parser.parse_args()


def resolve_output_path(input_path, output_path, camera_mode=False):
    if output_path is None:
        output_dir = './results'
        os.makedirs(output_dir, exist_ok=True)
        if camera_mode:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return os.path.join(output_dir, f'camera_{timestamp}.mp4')
        else:
            return os.path.join(output_dir, os.path.basename(input_path))
    else:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        return output_path


def is_video_file(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def mode_image_darkir(config, input_path, output_path):
    if not os.path.exists(input_path):
        print(f'[错误] 文件不存在: {input_path}')
        return

    print(f'[模型] DarkIR | 配置文件: {config}')
    model, resize = init_darkir(config)
    print(f'[成功] DarkIR 模型加载完成')

    output_path = resolve_output_path(input_path, output_path)
    result_path = darkir_enhance_single_image(model, input_path, output_path, resize)
    print(f'[完成] 图像增强完成: {result_path}')


def mode_batch_darkir(config, input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f'[错误] 不是文件夹: {input_dir}')
        return

    print(f'[模型] DarkIR | 配置文件: {config}')
    model, resize = init_darkir(config)
    print(f'[成功] DarkIR 模型加载完成')

    if output_dir is None:
        output_dir = './results'

    darkir_enhance_folder(model, input_dir, output_dir, resize)
    print(f'[完成] 批量图片增强完成')


def mode_video(args):
    input_path = args.input
    output_path = args.output
    model_choice = args.model

    if not os.path.exists(input_path):
        print(f'[错误] 文件不存在: {input_path}')
        return

    if not is_video_file(input_path):
        print(f'[错误] 不是视频文件: {input_path}')
        return

    output_path = resolve_output_path(input_path, output_path)

    if model_choice == 'darkir':
        print(f'[模型] DarkIR | 配置文件: {args.config}')
        model, resize = init_darkir(args.config)
        print(f'[成功] DarkIR 模型加载完成')
        result_path = darkir_enhance_video(model, input_path, output_path, resize)
    else:
        print(f'[模型] Zero-DCE')
        engine = create_inference_engine(model_path='models/Zero_DCE.pth')
        print(f'[成功] Zero-DCE 模型加载完成')
        result = engine.enhance_video(input_path, output_path)
        result_path = result['output_path']

    print(f'[完成] 视频增强完成: {result_path}')


def mode_video_batch(args):
    input_dir = args.input
    output_dir = args.output
    model_choice = args.model

    if not os.path.isdir(input_dir):
        print(f'[错误] 不是文件夹: {input_dir}')
        return

    if output_dir is None:
        output_dir = './results'

    if model_choice == 'darkir':
        print(f'[模型] DarkIR | 配置文件: {args.config}')
        model, resize = init_darkir(args.config)
        print(f'[成功] DarkIR 模型加载完成')
        darkir_enhance_videos_in_folder(model, input_dir, output_dir, resize)
    else:
        print(f'[模型] Zero-DCE')
        engine = create_inference_engine(model_path='models/Zero_DCE.pth')
        print(f'[成功] Zero-DCE 模型加载完成')

        video_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
        ]

        if not video_files:
            print(f'[警告] 未找到视频文件: {input_dir}')
            return

        print(f'[开始] 处理 {len(video_files)} 个视频...')
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            out_path = os.path.join(output_dir, video_name)
            engine.enhance_video(video_path, out_path)
            print(f'[完成] {video_name}')

    print('[完成] 批量视频增强完成')


def mode_camera(args):
    print(f'[模型] Zero-DCE')
    engine = create_inference_engine(model_path='models/Zero_DCE.pth')
    print(f'[成功] Zero-DCE 模型加载完成')

    save_path = None
    if args.save_video:
        save_path = resolve_output_path('', args.output, camera_mode=True)

    engine.enhance_camera(
        camera_id=args.camera_id,
        display=not args.no_display,
        save_video=args.save_video,
        save_path=save_path,
    )


def main():
    args = parse_args()

    print(f'[配置] 模式: {args.mode}')

    try:
        if args.mode == 'image':
            mode_image_darkir(args.config, args.input, args.output)
        elif args.mode == 'batch':
            mode_batch_darkir(args.config, args.input, args.output)
        elif args.mode == 'video':
            mode_video(args)
        elif args.mode == 'video_batch':
            mode_video_batch(args)
        elif args.mode == 'camera':
            mode_camera(args)

    except Exception as e:
        print(f'[错误] 运行失败: {str(e)}')
        sys.exit(1)

    print('[完成] 程序退出')


if __name__ == '__main__':
    main()
