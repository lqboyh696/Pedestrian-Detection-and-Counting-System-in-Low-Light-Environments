# 系统与文件操作相关模块
import os
import sys
import json
import uuid
import time
import base64
# 多线程与并发处理
import threading
from queue import Queue
import concurrent.futures  # 用于 WebSocket 的极速并发
# 日期时间处理
from datetime import datetime
# 数值计算与图像处理
import numpy as np
import cv2
import warnings

# Web 框架与实时通信
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit

# --- 1. 环境指路 ---
# 配置 CUDA 运行环境，将 CUDA 的 bin 目录加入 DLL 搜索路径和系统 PATH
cuda_bin_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin'
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
    os.environ['PATH'] = cuda_bin_path + ';' + os.environ['PATH']

# 关闭 ONNX Runtime 的 TensorRT 和日志输出，避免不必要的警告
os.environ['ORT_TENSORRT_UNAVAILABLE'] = '1'
os.environ['ORT_LOGGING_LEVEL'] = '3'
# 关闭 OpenCV 日志输出
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
# 忽略 Python 警告信息
warnings.filterwarnings("ignore")

# 深度学习框架
import torch

# 引入算法同学的增强库与智能滤镜：高光保护滤镜，避免增强后过曝
from image_filters import apply_highlight_protection

# 引入组员 A 提供的新版双模型增强库：DarkIR（暗光红外增强）和 Zero-DCE（零参考深度曲线估计增强）
import inference_DarkIR as darkir_api
from inference_Zero_DCE import ZeroDCEInference

# 引入组员 C 整合好的 YOLO 内存接口（0磁盘读写）
from yolov8_predict import YoloMemoryPredictor
# 监控画面表头绘制工具
from yolo_count_from_labels import draw_cctv_header
# ByteTrack 多目标跟踪与计数模块
from yolo_count_bytetrack_stable import (
    process_frame_memory_bytetrack,        # 基础 ByteTrack 跟踪
    process_frame_memory_full_lines,       # 划线计数模式
    process_frame_memory_zones_foot_region as process_frame_memory_zones,  # 区域计数模式（基于脚步区域）
)

# 创建 Flask 应用实例
app = Flask(__name__)
# 创建 SocketIO 实例，使用 threading 异步模式，允许跨域访问
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- 2. 系统配置 ---
# Flask Session 密钥，用于加密会话数据
app.secret_key = "campus_vision_unified_api_v4_parallel"
# 管理员密码
ADMIN_PASSWORD = "666666"
# 文件存储路径配置
UPLOAD_FOLDER = 'static/uploads'          # 用户上传的原始文件存放目录
RESULT_FOLDER = 'static/results'          # 处理后的结果文件存放目录
LIVE_FOLDER = 'static/live_records'       # 直播录像文件存放目录
DATA_FILE = 'results.json'                # 历史记录数据文件

# 确保各存储目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(LIVE_FOLDER, exist_ok=True)

# 全局状态管理器
# 管理并发录像任务，key 为 WebSocket 会话 ID，value 为录像状态字典
active_recorders = {}
# 管理各 WebSocket 用户的独立 YOLO 追踪状态（跟踪 ID、计数等）
client_states = {}
# 复用线程池，避免每帧处理时创建/销毁线程的开销
ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
# 视频上传处理进度共享状态，key 为文件名，value 为进度信息
processing_progress = {}

# --- 3. 初始化模型 ---
# 步骤1：加载 YOLOv8 行人检测模型
print("[1/3] 正在加载 YOLOv8 检测模型...")
detector = YoloMemoryPredictor('models/best.pt')

# 步骤2：加载 DarkIR 暗光增强模型
print("[2/3] 正在加载 DarkIR 模型...")
DARKIR_MODEL, darkir_resize = darkir_api.init_model('./options/DarkIR.yml')

# 步骤3：加载 Zero-DCE 低光增强模型
print("[3/3] 正在加载 Zero-DCE 模型...")
ZERO_DCE_ENGINE = ZeroDCEInference(model_path='models/Zero_DCE.pth', scale_factor=12)

print("AI 引擎就绪：所有模型已挂载完成！")

# --- GPU / 加速诊断 ---
# 打印系统硬件加速信息，帮助排查性能问题
print("\n" + "=" * 50)
print("  GPU / 加速诊断")
print("=" * 50)
# 输出 PyTorch 版本信息
print(f"  PyTorch 版本: {torch.__version__}")
# 检测 NVIDIA CUDA 是否可用
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    print(f"  当前设备: cuda:0")
    # 执行一次小规模张量运算测试 CUDA 是否正常工作
    try:
        dummy = torch.randn(1, 3, 256, 256, device='cuda')
        _ = torch.cuda.synchronize()
        print(f"  CUDA 测试: 张量运算正常")
    except Exception as e:
        print(f"  CUDA 测试: 失败 - {e}")
else:
    print("  WARNING: CUDA 不可用")
# MPS（Apple Metal Performance Shaders）诊断，检测 Apple Silicon GPU 加速是否可用
print(f"  MPS 可用: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    try:
        dummy = torch.randn(1, 3, 256, 256, device='mps')
        _ = torch.mps.synchronize()
        print(f"  MPS 测试: 张量运算正常（Apple GPU 加速可用）")
    except Exception as e:
        print(f"  MPS 测试: 失败 - {e}")
# 如果 CUDA 和 MPS 都不可用，发出警告
if not torch.cuda.is_available() and not torch.backends.mps.is_available():
    print("  WARNING: 无 GPU 加速可用，将使用 CPU（会很慢）")
# 输出 DarkIR 模型当前使用的计算设备
print(f"  DarkIR 设备: {darkir_api.device}")
print("=" * 50 + "\n")


# --- 4. 辅助函数与模型封装 ---

def save_result_to_json(filename, count, uid, duration, algo, record_type='upload', zone_id=None, zone_direction=None,
                        zone_count_in=None, zone_count_out=None, zone_count_cur=None,
                        line_id=None, count_mode='full'):
    """
    将处理结果保存到 JSON 文件中。
    参数说明：
    - filename: 结果文件名
    - count: 行人计数结果
    - uid: 用户唯一标识
    - duration: 处理耗时（秒）
    - algo: 使用的算法组合字符串
    - record_type: 记录类型（upload 上传处理 / live 直播录像）
    - zone_id: 区域 ID（区域计数模式时有效）
    - zone_direction: 方向（top_to_bottom / bottom_to_top 等）
    - zone_count_in/out/cur: 区域模式下的进入/离开/当前计数
    - line_id: 划线 ID（划线计数模式时有效）
    - count_mode: 计数模式（full 全图 / zone 区域 / line 划线）
    """
    # 读取已有数据（如果文件存在）
    data = {}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            pass

    # 如果文件中已有同名记录，先删除旧记录
    if filename in data:
        data.pop(filename)

    # 构建结果条目
    entry = {"count": count, "owner": uid, "duration": round(duration, 2), "algo": algo, "type": record_type}

    # 修复点 1：如果是区域模式，保存区域ID和方向
    if zone_id is not None:
        entry["zone_id"] = zone_id
        entry["zone_direction"] = zone_direction
        entry["zone_count_in"] = zone_count_in if zone_count_in is not None else count
        entry["zone_count_out"] = zone_count_out if zone_count_out is not None else 0
        entry["zone_count_cur"] = zone_count_cur if zone_count_cur is not None else count

    # 修复点 2：如果是划线模式，保存线ID的同时，务必也要保存方向
    if line_id:
        entry["line_id"] = line_id
        entry["zone_direction"] = zone_direction  # 就是漏了这一句

    # 保存计数模式
    if count_mode and count_mode != 'full':
        entry["count_mode"] = count_mode

    # 写入 JSON 文件
    data[filename] = entry
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def get_file_info(filename):
    """
    从 JSON 文件中读取指定文件的历史处理信息。
    返回包含计数、所属用户、处理耗时、算法等信息的字典。
    """
    default_info = {"count": "--", "owner": "unknown", "duration": "--", "algo": "未知", "type": "upload"}
    if not os.path.exists(DATA_FILE): return default_info
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f).get(filename, default_info)
        except:
            return default_info


def run_darkir_frame(frame):
    """
    封装调用组员 A 的 DarkIR 单帧处理。
    流程：尺寸调整（最高1080P） -> 转张量 -> 归一化 -> 模型增强 -> 转回数组 -> 还原原始尺寸
    参数 frame: BGR 格式的 numpy 数组图像
    返回: 增强后的 BGR 图像
    """
    try:
        H, W = frame.shape[:2]
        # 计算 1080P 等比例缩放后的目标尺寸
        new_H, new_W = darkir_api.calc_1080p_dims(H, W)

        # 尺寸限制，超1080P先缩小
        if new_H != H or new_W != W:
            process_frame = cv2.resize(frame, (new_W, new_H))
        else:
            process_frame = frame

        # 转换为 PyTorch 张量并传至指定设备
        tensor = darkir_api.array_to_tensor(process_frame).to(darkir_api.device)
        # 张量归一化
        tensor = darkir_api.normalize_tensor(tensor)
        # 送入 DarkIR 模型进行增强
        output_tensor = darkir_api.enhance_image(DARKIR_MODEL, tensor, resize=darkir_resize)
        # 将增强结果转回 numpy 数组
        result_img = darkir_api.tensor_to_array(output_tensor)

        # 为了不破坏坐标绑定体系，如果内部发生了缩放，缩放回原传入尺寸
        if result_img.shape[:2] != (H, W):
            result_img = cv2.resize(result_img, (W, H))

        return result_img
    except Exception as e:
        print(f"DarkIR 处理异常: {e}")
        return frame


def run_zero_dce_frame(frame):
    """
    极速优化版 Zero-DCE 单帧增强。
    将 NumPy 耗时操作全部转移至 GPU 张量运算，避免 CPU 阻塞。
    流程：尺寸对齐（12的倍数） -> GPU预处理（HWC->CHW） -> 推理 -> GPU后处理（clamp） -> 还原尺寸
    参数 frame: BGR 格式的 numpy 数组图像
    返回: 增强后的 BGR 图像
    """
    try:
        H, W = frame.shape[:2]

        # 1. 快速计算 12 的倍数尺寸（Zero-DCE 强制要求输入尺寸为 12 的倍数）
        new_H = (H // 12) * 12
        new_W = (W // 12) * 12
        if H != new_H or W != new_W:
            process_frame = cv2.resize(frame, (new_W, new_H))
        else:
            process_frame = frame

        # 2. 纯 GPU 预处理：直接把 uint8 传给 GPU，避免 CPU 阻塞
        tensor = torch.from_numpy(process_frame).to(ZERO_DCE_ENGINE.device, non_blocking=True)
        # 转置 HWC -> CHW, 增加 Batch 维度, 转 Float 并归一化（这一步在 GPU 上是瞬间完成的）
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # BGR 转 RGB（交换通道）
        tensor = tensor[:, [2, 1, 0], :, :]

        # 3. 网络推理（不计算梯度，节省显存）
        with torch.no_grad():
            enhanced_image, _ = ZERO_DCE_ENGINE.DCE_net(tensor)

        # 4. 纯 GPU 后处理：取代 CPU 的 np.clip，使用 PyTorch 的 clamp 方法
        enhanced_image = enhanced_image.clamp(0, 1) * 255.0
        # RGB 转 BGR（交换回通道顺序）
        enhanced_image = enhanced_image[:, [2, 1, 0], :, :]
        # 转回 uint8 并移至 CPU
        enhanced_np = enhanced_image.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

        # 5. 还原为前端发来的原始尺寸
        if enhanced_np.shape[:2] != (H, W):
            enhanced_np = cv2.resize(enhanced_np, (W, H))

        return enhanced_np
    except Exception as e:
        print(f"Zero-DCE 处理异常: {e}")
        return frame


# ====================================================================
# 视频文件处理：四线并行异步流水线（Fork-Join）增加抽帧逻辑
# 四线程设计：线程A-读取抽帧 -> 线程B-模型增强 -> 线程C-YOLO检测 -> 线程D-组装写入
# 线程B和线程C并行执行，大幅提升处理速度
# ====================================================================
def process_video_file(input_path, output_path, enable_enhance=True, enable_detect=True, zone_id=None,
                       zone_direction=None, zone_line_percent=12, count_mode='zone',
                       global_line_id='h1',
                       enhance_model='zero_dce', progress_key=None):
    """
    处理视频文件的主函数，采用四线程并行流水线架构。
    参数说明：
    - input_path: 输入视频路径
    - output_path: 输出视频路径
    - enable_enhance: 是否启用低光增强
    - enable_detect: 是否启用行人检测与计数
    - zone_id: 区域 ID
    - zone_direction: 计数方向
    - zone_line_percent: 区域线的纵向位置百分比
    - count_mode: 计数模式（zone 区域 / line 划线 / full 全图）
    - global_line_id: 划线 ID
    - enhance_model: 增强模型选择（darkir / zero_dce）
    - progress_key: 进度追踪的键名
    """
    # 打开输入视频文件
    cap = cv2.VideoCapture(input_path)

    # 强制 24 帧逻辑：如果原始帧率大于 24，则按比例抽帧，避免处理压力过大
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if original_fps > 24:
        target_fps = 24.0
        sample_interval = original_fps / 24.0
    else:
        target_fps = int(original_fps)
        sample_interval = 1.0

    # 获取原始视频尺寸
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用组员函数限制最高1080P，避免内存溢出
    target_h, target_w = darkir_api.calc_1080p_dims(orig_h, orig_w)

    # 精确预计算抽帧后的总帧数，防止线程永久挂起
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_total_frames = 0
    next_pos = 0.0
    for f in range(total_frames):
        if f >= int(next_pos):
            out_total_frames += 1
            next_pos += sample_interval

    # 使用 avc1 编码器输出 MP4 视频
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_w, target_h))

    # 1. 防爆环形缓冲字典（阅后即焚）：存储增强后的图像和检测框，按帧索引存取
    Image_Buffer_Dict = {}
    BBox_Buffer_Dict = {}

    # YOLO 跟踪状态初始化（用于全图计数和划线计数）
    yolo_state = {'prev_tracks': {}, 'count': 0, 'counted_ids': set()}
    # 区域计数跟踪状态初始化
    zone_state_pv = {'prev_tracks': {}, 'count': 0, 'counted_ids': set()}

    # 分发队列：最大容量 30，用于线程间解耦
    q_raw_dark = Queue(maxsize=30)   # 增强线程的输入队列
    q_raw_yolo = Queue(maxsize=30)   # 检测线程的输入队列

    # 线程 A：读取分发器（带抽帧跳帧）
    # 从视频中逐帧读取，按采样间隔抽帧，将帧同时分发给增强和检测队列
    def thread_read():
        idx = 0             # 抽帧后的索引
        frame_idx = 0       # 原始帧索引
        next_frame_pos = 0.0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 抽帧判定：仅当原始帧索引达到下一采样位置时才保留该帧
            if frame_idx >= int(next_frame_pos):
                if frame.shape[:2] != (target_h, target_w):
                    frame = cv2.resize(frame, (target_w, target_h))
                q_raw_dark.put((idx, frame))
                q_raw_yolo.put((idx, frame))
                idx += 1
                next_frame_pos += sample_interval

            frame_idx += 1
        # 发送结束信号
        q_raw_dark.put(None)
        q_raw_yolo.put(None)

    # 线程 B：模型增强层（根据前端传值智能切换 DarkIR/Zero-DCE）
    # 从增强队列取出帧，调用对应的增强模型，然后叠加高光保护滤镜
    def thread_enhance():
        while True:
            item = q_raw_dark.get()
            if item is None: break
            idx, frame = item
            if enable_enhance:
                orig = frame.copy()
                if enhance_model == 'darkir':
                    frame = run_darkir_frame(frame)
                else:
                    frame = run_zero_dce_frame(frame)
                # 高光保护：将增强后的暗部区域替换到原图上，避免过曝
                frame = apply_highlight_protection(orig, frame)
            Image_Buffer_Dict[idx] = frame

    # 线程 C：YOLO 语义抽取
    # 从检测队列取出帧，调用 YOLO 模型进行行人检测，结果存入检测框字典
    def thread_yolo():
        while True:
            item = q_raw_yolo.get()
            if item is None: break
            idx, frame = item
            boxes = detector.predict_frame(frame) if enable_detect else []
            BBox_Buffer_Dict[idx] = boxes

    # 线程 D：画师引擎
    # 轮询等待增强图像和检测框都就绪，然后组装画面、执行跟踪计数、写入输出视频
    def thread_assemble_and_write():
        nonlocal yolo_state, zone_state_pv
        target_idx = 0
        prev_time = time.time()

        # 等待抽帧后精准计算的数量
        while target_idx < out_total_frames:
            # 轮询双缓冲准备就绪
            if target_idx in Image_Buffer_Dict and target_idx in BBox_Buffer_Dict:
                e_frame = Image_Buffer_Dict.pop(target_idx)
                boxes = BBox_Buffer_Dict.pop(target_idx)

                # 计算实际处理帧率
                curr_time = time.time()
                fps_val = 1 / (curr_time - prev_time + 1e-5)
                prev_time = curr_time

                if enable_detect:
                    # 划线计数模式：使用 full_lines 进行基于检测线的计数
                    if zone_id is None and count_mode == 'line':
                        e_frame, yolo_state = process_frame_memory_full_lines(
                            frame=e_frame, curr_boxes=boxes, state=yolo_state,
                            line_id=global_line_id, direction=zone_direction,
                            fps_val=fps_val, is_video=True, show_fps=False
                        )
                    # 区域计数模式：使用 zones 进行基于区域的脚步检测计数
                    else:
                        e_frame, zone_state_pv = process_frame_memory_zones(
                            frame=e_frame, curr_boxes=boxes,
                            zone_id=zone_id, direction=zone_direction,
                            zone_state=zone_state_pv,
                            fps_val=fps_val, is_video=True,
                            line_percent=zone_line_percent, show_fps=False
                        )
                elif zone_id is not None:
                    # 不检测但仍需绘制区域标识的情况
                    e_frame, _ = process_frame_memory_zones(
                        frame=e_frame, curr_boxes=[],
                        zone_id=zone_id, direction=zone_direction,
                        zone_state={'prev_tracks': {}, 'count': 0, 'counted_ids': set()},
                        fps_val=fps_val, is_video=True,
                        line_percent=zone_line_percent, show_fps=False
                    )

                # 将组装好的帧写入输出视频
                out.write(e_frame)
                target_idx += 1
                # 每 30 帧输出一次进度
                if target_idx % 30 == 0:
                    print(f">>> 真并行处理进度: {target_idx}/{out_total_frames} 帧")
                # 更新进度共享状态供前端轮询
                if progress_key:
                    processing_progress[progress_key] = {"current": target_idx, "total": out_total_frames,
                                                         "percent": round(target_idx / out_total_frames * 100, 1)}
            else:
                # 缓冲未就绪，短暂休眠等待
                time.sleep(0.002)

    # 启动四驱马达：同时启动四个线程
    threads = [
        threading.Thread(target=thread_read),
        threading.Thread(target=thread_enhance),
        threading.Thread(target=thread_yolo),
        threading.Thread(target=thread_assemble_and_write)
    ]
    for t in threads: t.start()
    # 等待所有线程完成
    for t in threads: t.join()

    # 释放视频资源
    cap.release()
    out.release()

    # 标记处理完成
    if progress_key:
        processing_progress[progress_key] = {"current": out_total_frames, "total": out_total_frames,
                                             "percent": 100, "done": True}

    # 返回计数结果：区域模式返回进出双向统计，全图模式返回总计数
    if enable_detect and not (zone_id is None and count_mode == 'line'):
        return {
            'in': zone_state_pv.get('in_count', zone_state_pv['count']),
            'out': zone_state_pv.get('out_count', 0),
            'cur': zone_state_pv.get('current_count', zone_state_pv['count']),
        }
    return yolo_state['count'] if enable_detect else "--"


# --- 5. 路由逻辑 ---

# 首页路由：为每个用户分配唯一 ID，渲染主页
@app.route('/')
def index():
    if 'uid' not in session: session['uid'] = str(uuid.uuid4())
    return render_template('index.html', is_admin=session.get('is_admin', False))


# 管理员登录接口：验证密码，设置管理员会话
@app.route('/login', methods=['POST'])
def login():
    if request.get_json().get('password') == ADMIN_PASSWORD:
        session['is_admin'] = True
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 401


# 退出登录：清除管理员会话，重定向回首页
@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('index'))


# 文件上传处理接口：支持视频和图片，支持区域/划线/全图三种计数模式
@app.route('/upload', methods=['POST'])
def upload():
    # 获取上传的文件
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    # 解析文件名和扩展名，判断是视频还是图片
    original_name = file.filename
    file_ext = os.path.splitext(original_name)[1].lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']

    # 从表单中获取增强和检测的开关
    enable_enhance = request.form.get('enhance', 'true') == 'true'
    enable_detect = request.form.get('detect', 'true') == 'true'

    # 模型选择预留，视频默认 Zero-DCE，后期可在前端传参修改
    enhance_model = request.form.get('enhance_model', 'zero_dce')

    # 解析区域配置（JSON 格式字符串）
    raw_zone = request.form.get('zone', 'null')
    try:
        zone_info = json.loads(raw_zone)
        zone_id = zone_info.get('id') if zone_info else None
        count_mode = zone_info.get('mode', 'zone' if zone_id is not None else 'full') if zone_info else 'full'
        global_line_id = zone_info.get('line_id', 'h1') if zone_info else 'h1'
        zone_direction = zone_info.get('direction', 'top_to_bottom') if zone_info else None
        zone_line_percent = int(zone_info.get('line_percent', 12)) if zone_info else 12
        # 限制百分比在 1-50 之间
        zone_line_percent = max(1, min(50, zone_line_percent))
        if zone_id is not None:
            count_mode = 'zone'
    except (json.JSONDecodeError, TypeError, ValueError):
        # 解析失败时使用默认全图模式
        zone_id = None
        zone_direction = None
        zone_line_percent = 12
        count_mode = 'full'
        global_line_id = 'h1'

    # 构建算法名称字符串，用于前端展示
    algo_list = []
    if enable_enhance:
        algo_list.append("DarkIR" if not is_video or enhance_model == 'darkir' else "Zero-DCE")
    if enable_detect: algo_list.append("YOLOv8")
    algo_str = " + ".join(algo_list) if algo_list else "原图直出"

    # 保存原始上传文件
    raw_path = os.path.join(UPLOAD_FOLDER, original_name)
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
        except:
            pass
    file.save(raw_path)

    # 生成结果文件名（视频加 res_ 前缀，并强制 .mp4 扩展名）
    result_name = "res_" + (os.path.splitext(original_name)[0] + ".mp4" if is_video else original_name)
    result_path = os.path.join(RESULT_FOLDER, result_name)
    if os.path.exists(result_path):
        try:
            os.remove(result_path)
        except:
            pass

    # 获取当前用户 UID
    current_uid = session.get('uid')
    # 记录处理开始时间
    start_time = time.time()

    try:
        if is_video:
            # 视频处理：调用四线程并行流水线
            final_count = process_video_file(raw_path, result_path, enable_enhance, enable_detect,
                                             zone_id, zone_direction, zone_line_percent, count_mode,
                                             global_line_id,
                                             enhance_model=enhance_model,
                                             progress_key=result_name)
        else:
            # 单图处理模式：强制使用 DarkIR，并在进入流水线前统一最高 1080P
            img = cv2.imread(raw_path)
            orig_h, orig_w = img.shape[:2]
            new_h, new_w = darkir_api.calc_1080p_dims(orig_h, orig_w)

            if new_h != orig_h or new_w != orig_w:
                img = cv2.resize(img, (new_w, new_h))

            # 保留原图副本用于高光保护
            orig_img = img.copy()
            final_count = "--"
            boxes = []

            # YOLO 预测对齐原图
            if enable_detect:
                boxes = detector.predict_frame(orig_img)

            # 单图强制 DarkIR 增强
            if enable_enhance:
                img = run_darkir_frame(img)
                img = apply_highlight_protection(orig_img, img)

            # UI 渲染装配：根据计数模式调用对应的处理函数
            if enable_detect:
                if zone_id is not None:
                    # 区域计数模式
                    zone_state_init = {'prev_tracks': {}, 'count': 0, 'counted_ids': set()}
                    img, zone_state_result = process_frame_memory_zones(
                        frame=img, curr_boxes=boxes,
                        zone_id=zone_id, direction=zone_direction,
                        zone_state=zone_state_init,
                        fps_val=0, is_video=False, line_percent=zone_line_percent
                    )
                    final_count = zone_state_result['count']
                else:
                    # 全图计数模式（ByteTrack 跟踪）
                    image_state = {'prev_tracks': {}, 'count': 0, 'counted_ids': set()}
                    img, image_state = process_frame_memory_bytetrack(
                        frame=img, curr_boxes=boxes, state=image_state, fps_val=0, is_video=False
                    )
                    final_count = image_state['count']
            elif zone_id is not None:
                # 不检测但需要绘制区域标识
                img, _ = process_frame_memory_zones(
                    frame=img, curr_boxes=[],
                    zone_id=zone_id, direction=zone_direction,
                    zone_state={'prev_tracks': {}, 'count': 0, 'counted_ids': set()},
                    fps_val=0, is_video=False, line_percent=zone_line_percent
                )

            # 写入结果图片
            cv2.imwrite(result_path, img)

        # 计算处理耗时
        duration = time.time() - start_time
        # 保存结果到 JSON 历史记录
        if isinstance(final_count, dict):
            save_result_to_json(result_name, final_count['in'], current_uid, duration, algo_str, 'upload',
                                zone_id, zone_direction,
                                zone_count_in=final_count['in'],
                                zone_count_out=final_count['out'],
                                zone_count_cur=final_count['cur'],
                                line_id=global_line_id if count_mode == 'line' else None,
                                count_mode=count_mode)
        else:
            save_result_to_json(result_name, final_count, current_uid, duration, algo_str, 'upload',
                                zone_id, zone_direction,
                                line_id=global_line_id if count_mode == 'line' else None,
                                count_mode=count_mode)
        # 返回成功响应，重定向到结果页面
        return jsonify({"status": "success", "redirect": url_for('view_result', filename=result_name, fresh='1')})
    except Exception as e:
        return jsonify({"status": "error", "message": f"分析失败: {str(e)}"}), 500


# 结果查看页面：显示处理后的图片/视频及计数信息
@app.route('/view/<path:filename>')
def view_result(filename):
    is_admin = session.get('is_admin', False)
    # 从 JSON 文件获取历史处理信息
    info = get_file_info(filename)
    # 去掉 res_ 前缀还原原始文件名
    original_name = filename.replace("res_", "", 1)

    # 直播录像可能在 upload 目录没有原始文件
    is_live_record = (info.get('type') == 'live')
    if not is_live_record and not os.path.exists(os.path.join(UPLOAD_FOLDER, original_name)):
        original_name = filename

    # 判断是否为视频文件
    is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    zone_id = info.get('zone_id')
    zone_direction = info.get('zone_direction', 'top_to_bottom')
    return render_template('result.html', original_file=original_name, processed_file=filename,
                           filename=filename, count=info['count'], duration=info.get('duration', '--'),
                           algo=info.get('algo', '未知'), is_video=is_video, is_admin=is_admin,
                           is_live=is_live_record, zone_id=zone_id, zone_direction=zone_direction,
                           fresh=request.args.get('fresh'),
                           zone_count_cur=info.get('zone_count_cur', info['count']),
                           zone_count_in=info.get('zone_count_in', info['count']),
                           zone_count_out=info.get('zone_count_out', 0),
                           line_id=info.get('line_id', ''),
                           count_mode=info.get('count_mode', 'full'))


# 历史记录页面：管理员可见所有记录，普通用户只能看到自己的记录
@app.route('/history')
def history():
    is_admin = session.get('is_admin', False)
    current_uid = session.get('uid')
    if not os.path.exists(DATA_FILE): return render_template('history.html', upload_files=[], live_files=[])
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            data = {}

    # 分离上传文件和直播录像文件
    upload_files, live_files = [], []
    for fname, info in data.items():
        if is_admin or info.get('owner') == current_uid:
            if info.get('type') == 'live':
                live_files.append(fname)
            else:
                upload_files.append(fname)

    # 最新记录在前
    upload_files.reverse()
    live_files.reverse()
    return render_template('history.html', upload_files=upload_files, live_files=live_files)


# 删除历史记录接口
@app.route('/delete_history', methods=['POST'])
def delete_history():
    filename = request.get_json().get('filename')
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        if filename in records:
            del records[filename]
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False)
    return jsonify({"status": "success"})


# 前端轮询视频处理进度的接口
@app.route('/progress/<path:filename>')
def get_progress(filename):
    """前端轮询视频处理进度"""
    info = processing_progress.get(filename)
    if info:
        if info.get('done'):
            # 处理完成后清除进度信息
            processing_progress.pop(filename, None)
        return jsonify(info)
    return jsonify({"current": 0, "total": 0, "percent": 0})


# ====================================================================
# WebSocket 流媒体处理：使用 ThreadPoolExecutor 实现增强与检测并发，极速实时流
# 架构：前端通过 WebSocket 逐帧发送图像 -> 增强和检测线程池并行处理 -> 组装后推流回前端
# ====================================================================

def stop_and_save_recording(sid):
    """
    停止指定会话的录像并保存为 MP4 文件。
    处理流程：
    1. 如果还在缓冲阶段（未创建写入器），先用缓冲帧创建写入器
    2. 释放视频写入器
    3. 帧数太少（<10帧）的录像直接丢弃
    4. 帧数足够的录像保存到 LIVE_FOLDER 并写入 JSON 历史记录
    """
    if sid in active_recorders:
        rec = active_recorders[sid]
        # 如果还在缓冲阶段，用缓冲帧写出
        if rec['writer'] is None and rec.get('frame_buffer'):
            buf_frames = rec['frame_buffer']
            if len(buf_frames) >= 2:
                # 根据缓冲帧的时间戳计算真实帧率
                times = [t for _, t in buf_frames]
                intervals = [times[i] - times[i-1] for i in range(1, len(times))]
                real_fps = max(5.0, min(1.0 / (sum(intervals) / len(intervals)), 30.0))
            else:
                real_fps = 12.0
            # 创建视频写入器
            rec['writer'] = cv2.VideoWriter(rec['filepath'], cv2.VideoWriter_fourcc(*'avc1'),
                                             real_fps, (rec["target_w"], rec["target_h"]))
            # 将缓冲帧写入视频
            for buf_frame, _ in buf_frames:
                rec['writer'].write(cv2.resize(buf_frame, (rec["target_w"], rec["target_h"])))
            rec['frame_count'] = len(buf_frames)
            rec['frame_buffer'] = []
        # 释放视频写入器
        if rec['writer'] is not None:
            rec['writer'].release()
        # 计算录像时长
        duration = time.time() - rec['start_time']

        # 帧数太少（<10帧）视为无效录像，直接删除
        if rec['frame_count'] < 10:
            try:
                os.remove(rec['filepath'])
            except:
                pass
        else:
            # 从客户端状态获取计数结果
            c_state = client_states.get(sid, {})
            zone_id_save = c_state.get('zone_id')
            zone_direction_save = c_state.get('zone_direction')
            if c_state.get('zone_state'):
                zs = c_state['zone_state']
                final_count = {
                    'in': zs.get('in_count', zs.get('count', 0)),
                    'out': zs.get('out_count', 0),
                    'cur': zs.get('current_count', zs.get('count', 0))
                }
            else:
                final_count = c_state.get('count', 0)
            filename = os.path.basename(rec['filepath'])
            live_line_id = c_state.get('line_id', '')
            live_count_mode = c_state.get('count_mode', 'full')
            # 保存计数结果到 JSON 历史记录
            if isinstance(final_count, dict):
                save_result_to_json(filename, final_count['in'], rec['user_uid'], duration, rec['algo_str'], 'live',
                                    zone_id_save, zone_direction_save,
                                    zone_count_in=final_count['in'],
                                    zone_count_out=final_count['out'],
                                    zone_count_cur=final_count['cur'],
                                    line_id=live_line_id,
                                    count_mode=live_count_mode)
            else:
                save_result_to_json(filename, final_count, rec['user_uid'], duration, rec['algo_str'], 'live',
                                    zone_id_save, zone_direction_save,
                                    line_id=live_line_id,
                                    count_mode=live_count_mode)
            print(f">>> 直播已保存: {filename} (时长: {duration:.1f}s)")

        # 从活跃录像字典中移除
        del active_recorders[sid]


# WebSocket 实时流处理核心：接收前端发送的每一帧图像，进行增强和检测，推流返回
@socketio.on('stream_frame')
def handle_stream_frame(data):
    """
    处理前端通过 WebSocket 发送的实时视频帧。
    处理流程：
    1. 帧跳过防抖：如果上一帧还在处理中，直接丢弃当前帧
    2. 解析 Base64 图像为 OpenCV 格式
    3. 线程池并行执行增强和检测
    4. 根据计数模式调用对应的 ByteTrack 跟踪函数
    5. 如果开启录像，缓冲帧并写入 MP4 文件
    6. 将处理后的图像编码为 JPEG 推流回前端
    """
    sid = request.sid
    try:
        # 帧跳过防抖：如果上一帧还在处理中，直接丢弃当前帧，防止内存队列撑爆
        c_state = client_states.get(sid)
        if c_state and c_state.get('_busy', False):
            return

        # 从 WebSocket 消息中解析前端发送的参数
        img_data = data.get('image')
        enable_enhance = data.get('enhance', True)
        enable_detect = data.get('detect', True)
        quality = data.get('quality', '480p')
        is_recording = data.get('record', False)

        # 解析区域/划线配置
        raw_zone = data.get('zone')
        zone_id_ws = None
        zone_direction_ws = None
        count_mode_ws = 'full'
        global_line_id_ws = 'h1'
        line_percent_ws = 12
        if raw_zone and isinstance(raw_zone, dict):
            zone_id_ws = raw_zone.get('id')
            count_mode_ws = raw_zone.get('mode', 'zone' if zone_id_ws is not None else 'full')
            global_line_id_ws = raw_zone.get('line_id', 'h1')
            zone_direction_ws = raw_zone.get('direction', 'top_to_bottom')
            line_percent_ws = raw_zone.get('line_percent', 12)
            if zone_id_ws is not None:
                count_mode_ws = 'zone'

        # 将 Base64 编码的图像数据解码为 OpenCV BGR 格式
        header, encoded = img_data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 保留原图副本
        orig_frame = frame.copy()
        H_orig, W_orig = frame.shape[:2]

        # 初始化该 WebSocket 会话的客户端状态（首次连接时）
        if sid not in client_states:
            client_states[sid] = {'prev_tracks': {}, 'count': 0, 'counted_ids': set(),
                                  'prev_time': time.time(), 'zone_id': None,
                                  'zone_state': {'prev_tracks': {}, 'count': 0, 'counted_ids': set()},
                                  '_busy': False}

        c_state = client_states[sid]
        # 标记当前帧处理中，防止下一帧抢占
        c_state['_busy'] = True

        # 如果区域/计数模式发生了变化，重置相关状态
        prev_zone = c_state.get('zone_id')
        if (zone_id_ws != prev_zone
                or count_mode_ws != c_state.get('count_mode')
                or global_line_id_ws != c_state.get('global_line_id')
                or zone_direction_ws != c_state.get('zone_direction')):
            c_state['zone_id'] = zone_id_ws
            c_state['zone_direction'] = zone_direction_ws
            c_state['count_mode'] = count_mode_ws
            c_state['global_line_id'] = global_line_id_ws
            # 重置区域跟踪状态
            c_state['zone_state'] = {'prev_tracks': {}, 'count': 0, 'counted_ids': set()}

        # 计算实时帧率
        curr_t = time.time()
        fps = 1 / (curr_t - c_state['prev_time'] + 1e-5)
        c_state['prev_time'] = curr_t

        # 封装独立任务：增强任务（在线程池中执行）
        def task_enhance(img):
            if not enable_enhance: return img

            # 直接调用重写后的极速 GPU 版本 Zero-DCE
            enhanced = run_zero_dce_frame(img)

            # 核心警告：在实时流(直播)中，千万不要加 apply_highlight_protection
            # 这是导致 Web 端帧率比终端低几十帧的根本原因
            # 终端能跑 70FPS 是因为终端根本没跑这个滤镜
            return enhanced

        # 封装独立任务：检测任务（在线程池中执行）
        def task_detect(img):
            if not enable_detect: return []
            return detector.predict_frame(img)

        # 线程池双管齐下：增强和检测并行执行
        future_dark = ai_executor.submit(task_enhance, frame)
        future_yolo = ai_executor.submit(task_detect, frame)

        # 等待两个任务完成，获取结果
        final_frame = future_dark.result()
        boxes = future_yolo.result()

        count_display = "--"
        zone_display = None

        if enable_detect:
            # 划线计数模式
            if zone_id_ws is None and count_mode_ws == 'line':
                final_frame, c_state['zone_state'] = process_frame_memory_full_lines(
                    frame=final_frame, curr_boxes=boxes,
                    state=c_state['zone_state'],
                    line_id=global_line_id_ws, direction=zone_direction_ws,
                    fps_val=fps, is_video=True
                )
                count_display = str(c_state['zone_state'].get('count', 0))
            # 区域计数模式
            else:
                final_frame, c_state['zone_state'] = process_frame_memory_zones(
                    frame=final_frame, curr_boxes=boxes,
                    zone_id=zone_id_ws, direction=zone_direction_ws,
                    zone_state=c_state['zone_state'],
                    fps_val=fps, is_video=True, line_percent=line_percent_ws
                )
                # 构建计数显示字符串（CUR当前 / IN进入 / OUT离开）
                if ('current_count' in c_state['zone_state']
                        or 'in_count' in c_state['zone_state']
                        or 'out_count' in c_state['zone_state']):
                    count_display = (
                        f"CUR {c_state['zone_state'].get('current_count', 0)} / "
                        f"IN {c_state['zone_state'].get('in_count', 0)} / "
                        f"OUT {c_state['zone_state'].get('out_count', 0)}"
                    )
                else:
                    count_display = str(c_state['zone_state']['count'])
                if zone_id_ws is not None:
                    zone_display = zone_id_ws
        elif zone_id_ws is not None:
            # 不检测但需要绘制区域标识
            final_frame, _ = process_frame_memory_zones(
                frame=final_frame, curr_boxes=[],
                zone_id=zone_id_ws, direction=zone_direction_ws,
                zone_state={'prev_tracks': {}, 'count': 0, 'counted_ids': set()},
                fps_val=fps, is_video=True
            )
            zone_display = zone_id_ws
        else:
            # 纯增强模式：仅绘制 CCTV 画面表头
            draw_cctv_header(final_frame, "CAM 01", fps)
            count_display = "--"

        # 录像持久化逻辑
        if is_recording:
            # 首次开始录像时创建录像记录
            if sid not in active_recorders:
                filename = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.mp4"
                filepath = os.path.join(LIVE_FOLDER, filename)
                algo_str = ("Zero-DCE + " if enable_enhance else "") + ("YOLOv8" if enable_detect else "")
                active_recorders[sid] = {
                    "writer": None, "filepath": filepath, "start_time": time.time(),
                    "frame_count": 0, "algo_str": algo_str or "原图直出",
                    "user_uid": session.get('uid', 'unknown'), "target_w": W_orig, "target_h": H_orig,
                    "frame_buffer": []
                }

            rec = active_recorders[sid]
            if rec['writer'] is None:
                # 缓冲前 5 帧，计算真实处理帧率后再创建写入器
                rec['frame_buffer'].append((final_frame.copy(), time.time()))
                if len(rec['frame_buffer']) >= 5:
                    # 根据缓冲帧的时间间隔计算真实帧率
                    times = [t for _, t in rec['frame_buffer']]
                    intervals = [times[i] - times[i-1] for i in range(1, len(times))]
                    avg_interval = sum(intervals) / len(intervals)
                    real_fps = max(5.0, min(1.0 / avg_interval, 30.0))
                    # 创建 MP4 视频写入器
                    rec['writer'] = cv2.VideoWriter(rec['filepath'], cv2.VideoWriter_fourcc(*'avc1'),
                                                     real_fps, (W_orig, H_orig))
                    # 将缓冲帧写入视频文件
                    for buf_frame, _ in rec['frame_buffer']:
                        rec['writer'].write(cv2.resize(buf_frame, (rec["target_w"], rec["target_h"])))
                    rec['frame_count'] = len(rec['frame_buffer'])
                    rec['frame_buffer'] = []
            else:
                # 已创建写入器，直接将帧写入视频
                rec['writer'].write(cv2.resize(final_frame, (rec["target_w"], rec["target_h"])))
                rec['frame_count'] += 1
        else:
            # 停止录像时自动保存
            if sid in active_recorders:
                stop_and_save_recording(sid)

        # 推流前端展示：将处理后的图像编码为 JPEG，通过 WebSocket 发送给前端
        # 画质根据 quality 参数调整：720p 用 85，480p 用 70
        _, buffer = cv2.imencode('.jpg', final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85 if quality == '720p' else 70])
        response_data = {
            "status": "success",
            "image": "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8'),
            "count": count_display
        }
        if zone_display is not None:
            response_data["zone_id"] = zone_display

        emit('stream_result', response_data)
        # 解锁，迎接下一帧
        c_state['_busy'] = False
    except Exception as e:
        # 发生异常时也要解除忙状态，避免永久阻塞
        if sid in client_states:
            client_states[sid]['_busy'] = False
        emit('stream_result', {"status": "error", "message": str(e)})


# WebSocket 断开连接事件：自动保存录像并清理客户端状态
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    # 停止并保存正在进行的录像
    stop_and_save_recording(sid)
    # 清理该会话的客户端状态
    if sid in client_states:
        del client_states[sid]


# 应用启动入口
if __name__ == '__main__':
    # 启动 Flask-SocketIO 服务器，监听所有网卡，端口 5000
    socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
