"""
基于标签文件的人物检测计数脚本
- 读取 yolov8-predict.py 输出的 labels 文件夹中的框坐标
- 图片模式：统计人数 + 40% 透明计数面板
- 视频模式：越线计数 + 40% 透明计数面板 + CCTV抬头

用法：
    # ===== 步骤1：yolov8-predict.py 预测，输出坐标 =====
    # 图片模式
    默认：python3 yolov8-predict.py --mode image
    自定义：python3 yolov8-predict.py --mode image --input ./test --labels-dir my_labels --output runs/my_output

    # 视频模式
    默认：python3 yolov8-predict.py --mode video
    自定义：python3 yolov8-predict.py --mode video --input ./video_test/009.mp4 --labels-dir my_labels --output runs/my_output

    # ===== 步骤2：yolo-count-from-labels.py 计数 =====
    # 图片模式
    默认：python3 yolo-count-from-labels.py --mode image
    自定义：python3 yolo-count-from-labels.py --mode image --images ./test --labels runs/my_output/my_labels

    # 视频模式
    默认：python3 yolo-count-from-labels.py --mode video
    自定义：python3 yolo-count-from-labels.py --mode video --video ./video_test/009.mp4 --labels runs/my_output-2/my_labels
"""

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.files import increment_path
from pathlib import Path
import argparse
from datetime import datetime
import time


# ========== 默认配置（与 yolov8-predict.py 输出路径衔接）==========
ROOT = Path(__file__).resolve().parent
DEFAULT_MODE = 'video'
DEFAULT_LABELS = str(ROOT / 'runs' / 'predict_labels' / 'labels')
DEFAULT_IMAGES = str(ROOT / 'test')
DEFAULT_VIDEO = str(ROOT / 'video_test' / '009.mp4')
DEFAULT_OUTPUT = str(ROOT / 'runs' / 'counting_results_labels')
DEFAULT_CLASSES = [0]       # person
# ===========================


def parse_args():
    parser = argparse.ArgumentParser(description='基于标签文件的人物检测计数')
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE, choices=['image', 'video'])
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS, help='labels 文件夹路径')
    parser.add_argument('--images', type=str, default=DEFAULT_IMAGES, help='图片文件夹路径（image 模式）')
    parser.add_argument('--video', type=str, default=DEFAULT_VIDEO, help='视频路径（video 模式）')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--classes', nargs='+', type=int, default=DEFAULT_CLASSES)
    parser.add_argument('--line-ratio', type=float, default=0.5, help='越线位置比例（默认 0.5 = 画面正中）')
    return parser.parse_args()


def parse_label_file(label_path, target_classes):
    """解析单个标签文件，返回属于目标类别的框列表 [(x1,y1,x2,y2,conf,cls_name), ...]"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            cls_id = int(parts[0])
            cls_name = parts[1]
            conf = float(parts[2])
            x1 = float(parts[3])
            y1 = float(parts[4])
            x2 = float(parts[5])
            y2 = float(parts[6])
            if cls_id in target_classes:
                boxes.append((x1, y1, x2, y2, conf, cls_name))
    return boxes


def compute_centroid(x1, y1, x2, y2):
    """计算框的中心点"""
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def match_boxes(prev_tracks, curr_boxes, max_dist=80):
    """
    基于质心距离的贪心匹配，为当前帧框分配一致的 track_id。

    prev_tracks: dict {track_id: (cx, cy)}
    curr_boxes: list of (x1, y1, x2, y2, conf, cls_name)

    返回: dict {track_id: (x1,y1,x2,y2,cx,cy,conf,cls_name)}
    """
    if not prev_tracks:
        # 第一帧，给每个框分配新 ID
        matched = {}
        for i, box in enumerate(curr_boxes):
            x1, y1, x2, y2, conf, cls_name = box
            cx, cy = compute_centroid(x1, y1, x2, y2)
            matched[i] = (x1, y1, x2, y2, cx, cy, conf, cls_name)
        return matched

    if not curr_boxes:
        return {}

    # 计算当前框的质心
    curr_centroids = [compute_centroid(b[0], b[1], b[2], b[3]) for b in curr_boxes]

    # 构建距离矩阵
    prev_ids = list(prev_tracks.keys())
    prev_centroids = [prev_tracks[tid] for tid in prev_ids]

    matched = {}
    used_curr = set()
    used_prev = set()

    # 贪心匹配：按最小距离逐对匹配
    pairs = []
    for pi, pid in enumerate(prev_ids):
        pcx, pcy = prev_centroids[pi]
        for ci, (ccx, ccy) in enumerate(curr_centroids):
            dist = np.sqrt((pcx - ccx) ** 2 + (pcy - ccy) ** 2)
            if dist < max_dist:
                pairs.append((dist, pid, ci))

    pairs.sort(key=lambda x: x[0])
    for dist, pid, ci in pairs:
        if pid not in used_prev and ci not in used_curr:
            used_prev.add(pid)
            used_curr.add(ci)
            x1, y1, x2, y2, conf, cls_name = curr_boxes[ci]
            cx, cy = curr_centroids[ci]
            matched[pid] = (x1, y1, x2, y2, cx, cy, conf, cls_name)

    # 未匹配的新框分配新 ID
    new_id = max(prev_ids) + 1 if prev_ids else 0
    for ci in range(len(curr_boxes)):
        if ci not in used_curr:
            x1, y1, x2, y2, conf, cls_name = curr_boxes[ci]
            cx, cy = curr_centroids[ci]
            matched[new_id] = (x1, y1, x2, y2, cx, cy, conf, cls_name)
            new_id += 1

    return matched


# ==================== 绘图函数 ====================

def draw_count_panel(img, count, panel_alpha=0.4):
    """在左上角绘制 40% 透明计数面板"""
    h, w = img.shape[:2]
    panel_x1, panel_y1 = 10, 38
    panel_x2, panel_y2 = 240, 128

    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, panel_alpha, img, 1 - panel_alpha, 0, img)

    cv2.rectangle(img, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), 1)

    cv2.putText(img, "PERSON COUNT", (panel_x1 + 15, panel_y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, str(count), (panel_x1 + 15, panel_y1 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)


def draw_counting_line(img, line_y, line_alpha=0.3):
    """绘制半透明越线"""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.line(overlay, (0, line_y), (w, line_y), (0, 0, 255), 2)
    arrow_x = 60
    cv2.arrowedLine(overlay, (arrow_x, line_y - 20), (arrow_x, line_y + 20),
                    (0, 0, 255), 2, tipLength=0.4)
    cv2.addWeighted(overlay, line_alpha, img, 1 - line_alpha, 0, img)


def draw_cctv_header(img, camera_id, fps_val, line_alpha=0.4, show_fps=True):
    """绘制 CCTV 抬头（时间戳、摄像头ID，可选 FPS）"""
    h, w = img.shape[:2]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, line_alpha, img, 1 - line_alpha, 0, img)

    cv2.putText(img, camera_id, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, timestamp, (w - 400, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if show_fps:
        cv2.putText(img, f"FPS {fps_val:.1f}", (w - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ==================== 图片模式 ====================

def process_images(label_dir, image_dir, output_dir, args):
    """图片模式：读取标签文件 → 绘制检测框 + 40% 透明计数面板"""

    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    if image_dir.is_file():
        image_paths = [image_dir]
    elif image_dir.is_dir():
        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions])
        if not image_paths:
            print(f"错误：文件夹 {image_dir} 中没有图片")
            return
        print(f"找到 {len(image_paths)} 张图片")
    else:
        print(f"错误：路径无效 {image_dir}")
        return

    total_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 读取对应的标签文件
        label_path = label_dir / f"{img_path.stem}.txt"
        boxes = parse_label_file(label_path, args.classes)

        annotator = Annotator(img)
        count = 0
        for (x1, y1, x2, y2, conf, cls_name) in boxes:
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            label = f"{cls_name}"
            annotator.box_label([x1i, y1i, x2i, y2i], label, color=(200, 50, 50))
            count += 1

        draw_count_panel(img, count)
        total_count += count

        out_path = output_dir / f"{img_path.stem}_counted.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"[{idx}/{len(image_paths)}] {out_path.name} | {count} 人")

    print(f"\n图片处理完成！{len(image_paths)} 张 → {output_dir}")
    print(f"  总检测人次: {total_count}")


# ==================== 视频模式 ====================

def process_video(label_dir, video_path, output_dir, args):
    """视频模式：读取逐帧标签 → 质心追踪 → 边界入场计数 + 40% 透明面板"""

    label_dir = Path(label_dir)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = output_dir / f"{video_path.stem}_counted.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # 追踪状态
    prev_tracks = {}
    count = 0
    counted_ids = set()
    prev_time = 0

    print(f"视频: {video_path.name} | {width}x{height} | {fps:.1f}fps | 边界入场计数")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        current_time = time.time()
        fps_real = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        # CCTV 抬头
        draw_cctv_header(frame, "CAM 01", fps_real)

        # 读取当前帧标签
        label_path = label_dir / f"frame_{frame_idx:04d}.txt"
        curr_boxes = parse_label_file(label_path, args.classes)

        # 质心匹配追踪
        matched = match_boxes(prev_tracks, curr_boxes)

        annotator = Annotator(frame)
        new_prev_tracks = {}
        for track_id, (x1, y1, x2, y2, cx, cy, conf, cls_name) in matched.items():
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

            # 标准 YOLOv8 检测框 + 标签
            label = f"{cls_name}"
            annotator.box_label([x1i, y1i, x2i, y2i], label, color=(200, 50, 50))

            # 边界入场计数：每个新 track_id 首次出现即计数
            if track_id not in counted_ids:
                count += 1
                counted_ids.add(track_id)
                print(f"  Frame {frame_idx:04d} | ID {track_id} 入场! 累计: {count}")

            new_prev_tracks[track_id] = (cx, cy)

        prev_tracks = new_prev_tracks

        # 40% 透明计数面板
        draw_count_panel(frame, count)

        out.write(frame)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx:04d}/{total_frames} | count={count}")

        frame_idx += 1

    cap.release()
    out.release()

    print(f"\n视频处理完成！{frame_idx} 帧 → {out_path}")
    print(f"  入场计数人数: {count}")



# ==================== 主入口 ====================

def main():
    args = parse_args()
    print(f"模式: {args.mode}")
    print(f"标签目录: {args.labels}")

    label_dir = Path(args.labels)
    if not label_dir.exists():
        print(f"错误：标签目录不存在 {label_dir}")
        return

    output_dir = increment_path(Path(args.output), mkdir=True)

    if args.mode == 'image':
        image_dir = Path(args.images)
        if not image_dir.exists():
            print(f"错误：图片路径不存在 {image_dir}")
            return
        process_images(label_dir, image_dir, output_dir, args)
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"错误：视频路径不存在 {video_path}")
            return
        process_video(label_dir, video_path, output_dir, args)


# ====================================================================
# 🟢 新增：专为 Web 系统提供内存调用的 API 接口 (不读取 txt)
# ====================================================================
def process_frame_memory(frame, curr_boxes, prev_tracks, count, counted_ids,
                         fps_val=0.0, line_ratio=0.5, is_video=True, show_fps=True):
    """
    接收当前帧、检测框和追踪状态，在内存中完成追踪和 UI 绘制并返回更新后的状态。
    """
    height, width = frame.shape[:2]
    line_y = int(height * line_ratio)

    # 1. 质心匹配追踪
    matched = match_boxes(prev_tracks, curr_boxes)

    annotator = Annotator(frame)
    new_prev_tracks = {}

    # 2. 遍历渲染并计数
    for track_id, (x1, y1, x2, y2, cx, cy, conf, cls_name) in matched.items():
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        cxi, cyi = int(cx), int(cy)

        label = f"{cls_name}"
        annotator.box_label([x1i, y1i, x2i, y2i], label, color=(200, 50, 50))

        # 越线计数逻辑 (仅视频模式)
        if is_video and track_id in prev_tracks:
            prev_cy = prev_tracks[track_id][1]
            if prev_cy < line_y and cyi >= line_y:
                if track_id not in counted_ids:
                    count += 1
                    counted_ids.add(track_id)

        new_prev_tracks[track_id] = (cx, cy)

    if not is_video:
        count = len(curr_boxes)

    if is_video:
        draw_counting_line(frame, line_y)
        draw_cctv_header(frame, "CAM 01", fps_val, show_fps=show_fps)
    draw_count_panel(frame, count)

    return frame, new_prev_tracks, count, counted_ids

# ====================================================================
# 🟢 2x3 区域裁切放大计数功能 (Single Zone Crop + Boundary Counting)
# ====================================================================

def compute_zone_bounds(width, height, zone_id):
    """
    根据 2x3 网格计算区域像素边界。
    zone_id: 1-6, 按先行后列排列
    返回 (x1, y1, x2, y2)
    """
    col = (zone_id - 1) % 3
    row = (zone_id - 1) // 3
    x1 = col * width // 3
    x2 = (col + 1) * width // 3
    y1 = row * height // 2
    y2 = (row + 1) * height // 2
    return (x1, y1, x2, y2)


def process_frame_memory_zones(frame, curr_boxes, zone_id, direction, zone_state,
                               fps_val=0.0, is_video=True, line_percent=12, show_fps=True):
    """
    全帧输出 + 边界内侧 line_percent% 处越线计数。
    """
    height, width = frame.shape[:2]
    zx1, zy1, zx2, zy2 = compute_zone_bounds(width, height, zone_id)
    zone_w, zone_h = zx2 - zx1, zy2 - zy1
    LINE_RATIO = line_percent / 100.0

    # 计数线位置：边界内侧 20%
    if direction == 'top_to_bottom':
        line_pos = zy1 + zone_h * LINE_RATIO
        is_horizontal = True
    elif direction == 'bottom_to_top':
        line_pos = zy2 - zone_h * LINE_RATIO
        is_horizontal = True
    elif direction == 'left_to_right':
        line_pos = zx1 + zone_w * LINE_RATIO
        is_horizontal = False
    elif direction == 'right_to_left':
        line_pos = zx2 - zone_w * LINE_RATIO
        is_horizontal = False
    else:
        line_pos = 0
        is_horizontal = True

    # --- 1. 全帧追踪所有检测框 ---
    matched = match_boxes(zone_state['prev_tracks'], curr_boxes)

    # --- 2. 越线计数 (仅计区域内行人) ---
    new_prev_tracks = {}

    for track_id, (x1, y1, x2, y2, cx, cy, conf, cls_name) in matched.items():
        if is_video and track_id in zone_state['prev_tracks']:
            prev_cx, prev_cy = zone_state['prev_tracks'][track_id]

            in_zone_now = zx1 <= cx < zx2 and zy1 <= cy < zy2
            was_in_zone = zx1 <= prev_cx < zx2 and zy1 <= prev_cy < zy2

            if was_in_zone or in_zone_now:
                crossed = False
                if direction == 'top_to_bottom':
                    crossed = prev_cy < line_pos and cy >= line_pos
                elif direction == 'bottom_to_top':
                    crossed = prev_cy >= line_pos and cy < line_pos
                elif direction == 'left_to_right':
                    crossed = prev_cx < line_pos and cx >= line_pos
                elif direction == 'right_to_left':
                    crossed = prev_cx >= line_pos and cx < line_pos

                if crossed and track_id not in zone_state['counted_ids']:
                    zone_state['count'] += 1
                    zone_state['counted_ids'].add(track_id)

        new_prev_tracks[track_id] = (cx, cy)

    zone_state['prev_tracks'] = new_prev_tracks

    if not is_video:
        zone_state['count'] = sum(
            1 for box in curr_boxes
            if zx1 <= (box[0] + box[2]) / 2 < zx2 and zy1 <= (box[1] + box[3]) / 2 < zy2
        )

    # --- 3. 全帧绘制：所有检测框 ---
    annotator = Annotator(frame)
    for box in curr_boxes:
        bx1, by1, bx2, by2, conf, cls_name = box
        x1i, y1i, x2i, y2i = int(bx1), int(by1), int(bx2), int(by2)
        label = f"{cls_name}"
        annotator.box_label([x1i, y1i, x2i, y2i], label, color=(200, 50, 50))

    # --- 4. 叠加区域边界框 (半透明橙色) ---
    zone_color = (0, 180, 255)
    ov_zone = frame.copy()
    cv2.rectangle(ov_zone, (zx1, zy1), (zx2, zy2), zone_color, 2)
    zone_label = f"Z{zone_id}"
    cv2.putText(ov_zone, zone_label, (zx1 + 6, zy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
    cv2.addWeighted(ov_zone, 0.5, frame, 0.5, 0, frame)

    # --- 5. 计数线 (亮红色，在边界内侧 20% 处) ---
    ov_line = frame.copy()
    lp = int(line_pos)
    if is_horizontal:
        cv2.line(ov_line, (zx1, lp), (zx2, lp), (60, 60, 255), 3)
    else:
        cv2.line(ov_line, (lp, zy1), (lp, zy2), (60, 60, 255), 3)
    cv2.addWeighted(ov_line, 0.4, frame, 0.6, 0, frame)

    # --- 6. 绘制计数徽章 (左上角) ---
    badge_text = f"Z{zone_id} | {zone_state['count']}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ov = frame.copy()
    cv2.rectangle(ov, (6, 38), (tw + 18, th + 50), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, badge_text, (12, th + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if is_video:
        draw_cctv_header(frame, f"CAM 01-Z{zone_id}", fps_val, show_fps=show_fps)

    return frame, zone_state


if __name__ == '__main__':
    main()
