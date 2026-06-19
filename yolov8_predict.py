"""
YOLO 预测脚本（仅输出检测框坐标标签文件）
- 图片模式：批量推理图片，输出 labels/*.txt
- 视频模式：逐帧推理视频，输出 labels/frame_XXXX.txt

用法：
    # 图片模式
    python3 yolov8-predict.py --mode image --input ./test

    # 视频模式
    python3 yolov8-predict.py --mode video --input ./video_test/001.mp4

    # 自定义标签路径和输出
    python3 yolov8-predict.py --mode video \
        --input ./video_test/001.mp4 \
        --labels-dir my_labels \
        --output runs/my_results
"""

# yolov8_predict.py - YOLOv8 行人检测脚本
# 支持图片/视频两种模式，输出检测框坐标标签文件
# Web 系统通过 YoloMemoryPredictor 类实现内存级推理（零磁盘读写）

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.trackers import BYTETracker
from pathlib import Path
import argparse
import torch
import numpy as np
from types import SimpleNamespace


# ========== 设备自动检测 ==========
if torch.cuda.is_available():
    DETECT_DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DETECT_DEVICE = 'mps'
else:
    DETECT_DEVICE = 'cpu'

print(f"[YOLO] 检测设备: {DETECT_DEVICE.upper()}{' (' + torch.cuda.get_device_name(0) + ')' if DETECT_DEVICE == 'cuda' else ''}")


# ========== 默认配置 ==========
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = str(ROOT / 'best.pt')
DEFAULT_MODE = 'video'
DEFAULT_INPUT = str(ROOT / 'video_test' / '009.mp4')
DEFAULT_OUTPUT = str(ROOT / 'runs' / 'predict_labels')
DEFAULT_CLASSES = [0]       # person
DEFAULT_CONF = 0.2          # 置信度阈值
DEFAULT_IOU = 0.45          # IoU 阈值
# ===========================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def parse_args():
    '''命令行参数解析'''
    parser = argparse.ArgumentParser(description='YOLO 预测脚本（仅输出检测框坐标）')
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE, choices=['image', 'video'])
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--classes', nargs='+', type=int, default=DEFAULT_CLASSES)
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF)
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU)
    parser.add_argument('--labels-dir', type=str, default=None, help='标签子文件夹名称或绝对路径（默认 labels）')
    return parser.parse_args()


def save_labels(label_dir, identifier, boxes, names):
    """将预测框坐标保存为 txt 标签文件"""
    label_path = label_dir / f"{identifier}.txt"
    with open(label_path, 'w') as f:
        if boxes is None or len(boxes) == 0:
            return
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = names.get(cls_id, str(cls_id))
            # 格式: cls_id cls_name confidence x1 y1 x2 y2
            f.write(f"{cls_id} {cls_name} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")


# ==================== 图片模式 ====================

def process_images(model, input_path, label_dir, args):
    """批量推理图片，仅输出 labels/*.txt 标签文件"""

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if not image_paths:
            print(f"错误：文件夹 {input_path} 中没有图片")
            return
        print(f"找到 {len(image_paths)} 张图片，开始批量处理...")
    else:
        print(f"错误：路径无效 {input_path}")
        return

    total_detections = 0
    for idx, img_path in enumerate(image_paths, 1):
        results = model.predict(
            source=str(img_path),
            device='mps',
            classes=args.classes,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )
        boxes = results[0].boxes

        stem = img_path.stem
        save_labels(label_dir, stem, boxes, model.names)

        count = len(boxes) if boxes is not None else 0
        total_detections += count
        print(f"[{idx}/{len(image_paths)}] {img_path.name} | {count} 个检测 → labels/{stem}.txt")

    print(f"\n图片处理完成！{len(image_paths)} 张，{total_detections} 个检测框 → {label_dir}")


# ==================== 视频模式 ====================

def process_video(model, input_path, label_dir, args):
    """逐帧推理视频，仅输出 labels/frame_XXXX.txt 标签文件"""

    results = model.predict(
        source=str(input_path),
        stream=True,           # 流式逐帧处理，节省内存
        device='mps',
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    frame_idx = 0
    total_detections = 0

    for result in results:
        boxes = result.boxes
        save_labels(label_dir, f"frame_{frame_idx:04d}", boxes, model.names)

        count = len(boxes) if boxes is not None else 0
        total_detections += count

        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx:04d} | {count} 个检测")

        frame_idx += 1

    print(f"\n视频处理完成！{frame_idx} 帧，{total_detections} 个检测框 → {label_dir}")


# ==================== 主入口 ====================

def main():
    args = parse_args()

    model = YOLO(args.model, task='detect')
    print(f"模型: {args.model}")
    print(f"模式: {args.mode}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：路径不存在 {input_path}")
        return

    # 自动递增输出目录名
    output_dir = increment_path(Path(args.output), mkdir=True)

    if args.labels_dir:
        label_path = Path(args.labels_dir)
        label_dir = label_path if label_path.is_absolute() else output_dir / args.labels_dir
    else:
        label_dir = output_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == 'image':
        process_images(model, input_path, label_dir, args)
    else:
        process_video(model, input_path, label_dir, args)


# ====================================================================
# 新增：专为 Web 系统提供内存调用的 API 接口 (不写入磁盘 txt)
# ====================================================================
class YoloMemoryPredictor:
    """YOLO 内存预测器：接收 numpy 帧，直接返回检测框列表，零磁盘读写"""
    def __init__(self, model_path, classes=DEFAULT_CLASSES, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
        self.model = YOLO(model_path, task='detect')
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.names = self.model.names  # 类别名映射

    def predict_frame(self, frame):
        """输入内存图像 numpy 数组，直接返回预测框坐标列表 [(x1,y1,x2,y2,conf,cls_name), ...]"""
        results = self.model.predict(
            source=frame, device=DETECT_DEVICE, classes=self.classes,
            conf=self.conf, iou=self.iou, verbose=False
        )

        boxes_data = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = self.names.get(cls_id, str(cls_id))
                boxes_data.append((x1, y1, x2, y2, conf, cls_name))

        return boxes_data


# ====================================================================
# ByteTrack 会话：每个 WebSocket 客户端 / 文件上传任务独立实例
# ====================================================================
class ByteTrackSession:
    """
    轻量级 ByteTrack 封装，为单会话提供稳定追踪 ID。
    YOLO 模型全客户端共享，ByteTrack 每会话独立 -- 避免 ID 跨客户端冲突。
    """

    def __init__(self):
        self.tracker = BYTETracker(
            args=SimpleNamespace(
                track_low_thresh=0.05,
                track_high_thresh=0.15,
                new_track_thresh=0.2,
                track_buffer=30,
                match_thresh=0.2,
                fuse_score=False,
            ),
        )

    def update(self, boxes, frame_shape):
        """
        输入 YOLO 检测框，输出带稳定 track_id 的检测框。
        Returns: [(x1,y1,x2,y2,conf,cls_name,track_id), ...]
        """
        if boxes:
            xyxy = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=np.float32)
            conf = np.array([b[4] for b in boxes], dtype=np.float32)
            cls_arr = np.zeros(len(boxes), dtype=np.float32)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls_arr = np.empty((0,), dtype=np.float32)

        # 构造兼容 ultralytics BYTETracker 的伪 Results 对象
        class FakeResults:
            def __init__(self, xyxy_arr, conf_arr, cls_arr):
                self._xyxy = xyxy_arr
                self.xywh = self._xyxy2xywh(xyxy_arr)
                self.conf = conf_arr
                self.cls = cls_arr

            @staticmethod
            def _xyxy2xywh(x):
                """[x1,y1,x2,y2] -> [cx,cy,w,h]"""
                r = np.zeros_like(x)
                r[:, 0] = (x[:, 0] + x[:, 2]) / 2
                r[:, 1] = (x[:, 1] + x[:, 3]) / 2
                r[:, 2] = x[:, 2] - x[:, 0]
                r[:, 3] = x[:, 3] - x[:, 1]
                return r

            def __getitem__(self, idx):
                """支持布尔/整数索引，BYTETracker 内部会 results[inds] 过滤"""
                return FakeResults(self._xyxy[idx], self.conf[idx], self.cls[idx])

            def __len__(self):
                return len(self._xyxy)

        results = FakeResults(xyxy, conf, cls_arr)
        img = np.zeros((*frame_shape, 3), dtype=np.uint8)
        tracked = self.tracker.update(results, img)

        result = []
        for t in tracked:
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            tid = int(t[4])
            conf_val = float(t[5])
            result.append((x1, y1, x2, y2, conf_val, 'person', tid))

        return result

    def reset(self):
        """重置追踪器状态（切换区域/视频时调用）"""
        self.tracker.reset()


if __name__ == '__main__':
    main()
