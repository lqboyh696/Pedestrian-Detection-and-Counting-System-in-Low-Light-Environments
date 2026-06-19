import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import os
import sys
import time
import model
import numpy as np
from torchvision import transforms
import cv2
import glob
import threading
import queue
from typing import Optional, Tuple, List, Union
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ZeroDCEInference:
    """Zero-DCE 图像增强推理类"""

    def __init__(self, model_path: str = 'model/Zero_DCE.pth',
                 scale_factor: int = 12,
                 use_fp16: bool = False):
        """初始化推理模型"""
        self.scale_factor = scale_factor
        self.use_fp16 = use_fp16
        # 🍎 设备自动选择：CUDA > MPS（Apple GPU） > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f'[初始化] 模型: {model_path}, 设备: {self.device}')

        self.DCE_net = self._load_model(model_path)
        cudnn.benchmark = True

    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型权重"""
        DCE_net = model.enhance_net_nopool(self.scale_factor).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

        DCE_net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        if self.use_fp16:
            DCE_net = DCE_net.half()

        DCE_net.eval()
        return DCE_net

    def _sync(self):
        """安全同步：根据设备类型调用对应同步方法"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像：BGR转RGB，归一化，裁剪为scale_factor倍数"""
        if image is None:
            raise ValueError('图像读取失败')

        data_lowlight = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_lowlight = (data_lowlight.astype(np.float32) / 255.0)

        h = (data_lowlight.shape[0] // self.scale_factor) * self.scale_factor
        w = (data_lowlight.shape[1] // self.scale_factor) * self.scale_factor
        data_lowlight = data_lowlight[0:h, 0:w, :]
        data_lowlight = data_lowlight.transpose(2, 0, 1)
        tensor = torch.from_numpy(data_lowlight).float().unsqueeze(0)

        if self.use_fp16:
            tensor = tensor.half()

        return tensor

    def enhance_single_image(self, image_path: str, save_path: Optional[str] = None) -> Tuple[float, Optional[str]]:
        """增强单张图像，返回(耗时秒, 保存路径)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'图像文件不存在: {image_path}')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'无法读取图像: {image_path}')

        tensor = self.preprocess_image(image).to(self.device, non_blocking=True)

        start_time = time.time()
        with torch.no_grad():
            enhanced_image, _ = self.DCE_net(tensor)

        if self.use_fp16:
            enhanced_image = enhanced_image.float()

        elapsed = time.time() - start_time

        if save_path is None:
            save_dir = 'results/'
            os.makedirs(save_dir, exist_ok=True)
            image_name = os.path.basename(image_path)
            save_path = os.path.join(save_dir, image_name)
        else:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        torchvision.utils.save_image(enhanced_image, save_path)

        return elapsed, save_path

    def enhance_batch_images(self, input_dir: str, output_dir: Optional[str] = None,
                             recursive: bool = False) -> dict:
        """批量增强文件夹中的图像（多线程优化）"""
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f'输入目录不存在: {input_dir}')

        if output_dir is None:
            output_dir = 'results/'
        os.makedirs(output_dir, exist_ok=True)

        # 收集所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp',
                            '*.JPG', '*.JPEG', '*.PNG', '*.BMP']

        test_list = []
        for ext in image_extensions:
            if recursive:
                test_list.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
            else:
                test_list.extend(glob.glob(os.path.join(input_dir, ext)))

        test_list = list(set(test_list))
        test_list.sort()

        if len(test_list) == 0:
            print(f'[警告] 未找到图片文件')
            return {'total': 0, 'success': 0, 'failed': 0}

        print(f'[开始] 处理 {len(test_list)} 张图片...')

        SENTINEL = object()
        preprocess_queue = queue.Queue(maxsize=4)
        save_queue = queue.Queue(maxsize=8)

        t_preprocess = threading.Thread(
            target=self._preprocess_worker,
            args=(test_list, preprocess_queue, SENTINEL),
            daemon=True
        )

        t_save = threading.Thread(
            target=self._save_worker,
            args=(save_queue, SENTINEL),
            daemon=True
        )

        overall_start = time.time()
        t_preprocess.start()
        t_save.start()

        total_images, all_timings, save_time_holders = self._inference_loop(
            preprocess_queue, save_queue, len(test_list), output_dir, SENTINEL
        )

        t_preprocess.join()
        save_queue.put(SENTINEL)
        t_save.join()

        overall_time = time.time() - overall_start
        stats = self._compute_statistics(total_images, all_timings, save_time_holders, overall_time)

        return stats

    def enhance_video(self, video_path: str, output_path: Optional[str] = None, 
                      display: bool = False) -> dict:
        """增强视频文件（自动压缩到1080P）"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'视频文件不存在: {video_path}')
        
        if output_path is None:
            output_dir = 'results/'
            os.makedirs(output_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f'{video_name}_enhanced.mp4')
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f'无法打开视频: {video_path}')
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算输出分辨率（短边限制为1080P）
        shorter = min(height, width)
        if shorter > 1080:
            scale = 1080.0 / shorter
            out_width = int(round(width * scale))
            out_height = int(round(height * scale))
            need_resize = True
            print(f'[压缩] {os.path.basename(video_path)}: {width}x{height} -> {out_width}x{out_height}')
        else:
            out_width = width
            out_height = height
            need_resize = False
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        print(f'[开始] 处理视频: {os.path.basename(video_path)} ({total_frames}帧)')
        
        frame_count = 0
        total_time = 0
        success_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 如果需要压缩，先resize
                if need_resize:
                    frame = cv2.resize(frame, (out_width, out_height))
                
                tensor = self.preprocess_image(frame).to(self.device, non_blocking=True)
                
                start_time = time.time()
                with torch.no_grad():
                    enhanced_image, _ = self.DCE_net(tensor)
                
                if self.use_fp16:
                    enhanced_image = enhanced_image.float()
                
                enhanced_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                out.write(enhanced_bgr)
                success_count += 1
                
                if display:
                    cv2.imshow('Enhanced', enhanced_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f'[进度] {frame_count}/{total_frames} 帧')
                
            except Exception as e:
                continue
        
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
        
        avg_fps = success_count / total_time if total_time > 0 else 0
        print(f'[完成] {success_count}/{total_frames} 帧, {avg_fps:.2f} FPS')
        
        return {
            'total_frames': total_frames,
            'success_frames': success_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'output_path': output_path
        }

    def enhance_camera(self, camera_id: int = 0, display: bool = True,
                       save_video: bool = False, save_path: Optional[str] = None) -> dict:
        """摄像头实时增强（1080P@24fps）"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f'无法打开摄像头 {camera_id}')

        # 设置摄像头参数为1080P
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 24)

        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 24

        # 计算实际处理后的尺寸（scale_factor倍数）
        processed_width = (raw_width // self.scale_factor) * self.scale_factor
        processed_height = (raw_height // self.scale_factor) * self.scale_factor

        writer = None
        if save_video:
            if save_path is None:
                save_dir = 'results/'
                os.makedirs(save_dir, exist_ok=True)
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_dir, f'camera_{timestamp}.mp4')

            # 使用处理后的尺寸初始化VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (processed_width, processed_height))

            if not writer.isOpened():
                print(f'[错误] 视频写入器初始化失败')
                writer = None
            else:
                print(f'[保存] 分辨率: {processed_width}x{processed_height}')

        print(f'[开始] 摄像头增强 {raw_width}x{raw_height}@{fps}fps (按q退出)')

        frame_count = 0
        total_time = 0
        saved_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                tensor = self.preprocess_image(frame).to(self.device, non_blocking=True)

                start_time = time.time()
                with torch.no_grad():
                    enhanced_image, _ = self.DCE_net(tensor)

                if self.use_fp16:
                    enhanced_image = enhanced_image.float()

                enhanced_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
                enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

                elapsed = time.time() - start_time
                total_time += elapsed
                frame_count += 1

                # 保存视频
                if writer is not None:
                    try:
                        writer.write(enhanced_bgr)
                        saved_frames += 1
                    except Exception:
                        pass

                # 显示画面
                if display:
                    cv2.putText(enhanced_bgr, f'FPS: {1 / elapsed:.1f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Enhanced', enhanced_bgr)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # 定期输出状态
                if frame_count % 120 == 0:
                    avg_fps = frame_count / total_time
                    msg = f'[状态] {frame_count}帧, {avg_fps:.1f} FPS'
                    if writer:
                        msg += f', 已保存{saved_frames}帧'
                    print(msg)

        except KeyboardInterrupt:
            print('\n[中断]')
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f'[完成] {frame_count}帧, {avg_fps:.1f} FPS, 保存{saved_frames}帧')

        return {
            'total_frames': frame_count,
            'saved_frames': saved_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'saved_video': save_path if save_video and writer is not None else None
        }

    # ========== 内部辅助方法 ==========

    def _preprocess_worker(self, image_paths, preprocess_queue, sentinel):
        """预处理线程Worker"""
        for image_path in image_paths:
            timing = {}
            try:
                t0 = time.time()
                data_lowlight = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if data_lowlight is None:
                    raise ValueError(f'无法读取图像')
                data_lowlight = cv2.cvtColor(data_lowlight, cv2.COLOR_BGR2RGB)
                timing['read'] = time.time() - t0

                t1 = time.time()
                data_lowlight = (data_lowlight.astype(np.float32) / 255.0)
                h = (data_lowlight.shape[0] // self.scale_factor) * self.scale_factor
                w = (data_lowlight.shape[1] // self.scale_factor) * self.scale_factor
                data_lowlight = data_lowlight[0:h, 0:w, :]
                data_lowlight = data_lowlight.transpose(2, 0, 1)
                tensor = torch.from_numpy(data_lowlight).float().unsqueeze(0).pin_memory()
                if self.use_fp16:
                    tensor = tensor.half()
                timing['preprocess'] = time.time() - t1

                preprocess_queue.put((image_path, tensor, timing))

            except Exception as e:
                preprocess_queue.put((image_path, None, {}))

        preprocess_queue.put(sentinel)

    def _save_worker(self, save_queue, sentinel):
        """保存线程Worker"""
        while True:
            item = save_queue.get()
            if item is sentinel:
                break
            enhanced_image, result_path, use_fp16, save_time_holder = item
            try:
                t0 = time.time()
                if use_fp16:
                    enhanced_image = enhanced_image.float()
                torchvision.utils.save_image(enhanced_image, result_path)
                save_time_holder.append(time.time() - t0)
            except Exception:
                save_time_holder.append(0.0)

    def _inference_loop(self, preprocess_queue, save_queue, total_count, result_dir, sentinel):
        """主推理循环"""
        finished = 0
        all_timings = []
        save_time_holders = []

        while True:
            item = preprocess_queue.get()
            if item is sentinel:
                break

            image_path, tensor, pre_timing = item
            image_name = os.path.basename(image_path)

            if tensor is None:
                continue

            try:
                t0 = time.time()
                tensor = tensor.to(self.device, non_blocking=True)
                self._sync()
                transfer_time = time.time() - t0

                with torch.no_grad():
                    t1 = time.time()
                    enhanced_image, params_maps = self.DCE_net(tensor)
                    self._sync()
                    infer_time = time.time() - t1

                result_path = os.path.join(result_dir, image_name)
                save_time_holder = []
                save_queue.put((
                    enhanced_image.clone(),
                    result_path,
                    self.use_fp16,
                    save_time_holder
                ))
                save_time_holders.append(save_time_holder)

                finished += 1
                timing = {
                    **pre_timing,
                    'transfer': transfer_time,
                    'infer': infer_time,
                }
                all_timings.append(timing)

                # 每50张或最后一张打印进度
                if finished % 50 == 0 or finished == total_count:
                    print(f'[进度] {finished}/{total_count}')

            except Exception:
                pass

        return finished, all_timings, save_time_holders

    def _compute_statistics(self, total_images, all_timings, save_time_holders, overall_time):
        """计算统计信息"""
        if total_images == 0:
            return {'total': 0, 'success': 0, 'failed': 0}

        reads = [t.get('read', 0) for t in all_timings]
        preprocs = [t.get('preprocess', 0) for t in all_timings]
        transfers = [t.get('transfer', 0) for t in all_timings]
        infers = [t.get('infer', 0) for t in all_timings]
        saves = [h[0] for h in save_time_holders if h]

        def avg_ms(lst):
            return sum(lst) / len(lst) * 1000 if lst else 0.0

        per_frame_totals = [
            reads[i] + preprocs[i] + transfers[i] + infers[i] + (saves[i] if i < len(saves) else 0)
            for i in range(total_images)
        ]

        print('\n' + '=' * 60)
        print(f'总数: {total_images} | 总时间: {overall_time:.2f}s | FPS: {total_images / overall_time:.2f}')
        print(f'平均: 读取{avg_ms(reads):.1f}ms 预处理{avg_ms(preprocs):.1f}ms '
              f'传输{avg_ms(transfers):.1f}ms 推理{avg_ms(infers):.1f}ms 保存{avg_ms(saves):.1f}ms')
        print(f'端到端: {avg_ms(per_frame_totals):.1f}ms/张')
        print('=' * 60)

        return {
            'total': total_images,
            'success': total_images,
            'failed': 0,
            'total_time': overall_time,
            'avg_fps': total_images / overall_time,
            'avg_read_ms': avg_ms(reads),
            'avg_preprocess_ms': avg_ms(preprocs),
            'avg_transfer_ms': avg_ms(transfers),
            'avg_infer_ms': avg_ms(infers),
            'avg_save_ms': avg_ms(saves),
            'avg_total_ms': avg_ms(per_frame_totals)
        }

def create_inference_engine(model_path='model/Zero_DCE.pth', scale_factor=12, use_fp16=False):
    """工厂函数：创建推理引擎实例"""
    return ZeroDCEInference(model_path=model_path, scale_factor=scale_factor, use_fp16=use_fp16)
