# Pedestrian Detection and Counting System in Low-Light Environments

低照度环境下基于深度学习的行人检测与计数系统。集成 **DarkIR / Zero-DCE 低光增强** + **YOLOv8 目标检测** + **ByteTrack 多目标跟踪**，通过 Flask + SocketIO 提供完整的 Web 交互界面，并支持 CLI 命令行批量处理。

> **专业综合设计课程作业** | 四人团队协作项目

**Contact**: <lqboyh@gmail.com>

***

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Download Model Weights](#download-model-weights)
  - [Setup (Windows)](#setup-windows)
  - [Setup (macOS)](#setup-macos)
  - [CUDA Path Configuration](#cuda-path-configuration)
- [Usage](#usage)
  - [Web Application](#web-application)
  - [Command Line Interface](#command-line-interface)
- [Configuration Guide](#configuration-guide)
- [Methodology](#methodology)
  - [1. Low-Light Image Enhancement](#1-low-light-image-enhancement)
  - [2. Pedestrian Detection (YOLOv8)](#2-pedestrian-detection-yolov8)
  - [3. Multi-Object Tracking (ByteTrack)](#3-multi-object-tracking-bytetrack)
  - [4. Three Counting Modes](#4-three-counting-modes)
  - [5. Post-Processing: Highlight Protection](#5-post-processing-highlight-protection)
- [System Pipeline](#system-pipeline)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

***

## Key Features

- **Dual Low-Light Enhancement**: DarkIR (frequency-domain U-Net, \~15-20 GMACs) for superior image quality; Zero-DCE (lightweight depthwise CNN, \~4-5 GMACs) for real-time video streaming
- **Three Counting Modes**:
  - **Full-image counting**: ByteTrack-based line crossing across the entire frame
  - **Zone-based counting**: 6-zone grid (2x3) with foot-region IN/OUT directional tracking
  - **Line-crossing counting**: Configurable horizontal/vertical counting lines (H1/V1/V2)
- **4-Thread Parallel Video Pipeline**: Read → Enhance & Detect (parallel branches) → Assemble & Track, with adaptive frame sampling (≤24 FPS) and auto 1080p downscaling
- **Real-time WebSocket Streaming**: Low-latency frame push with concurrent enhance + detect via `ThreadPoolExecutor`; frame-skip debounce prevents memory queue overflow
- **Live Recording**: Auto-buffered real-time stream recording with computed FPS from actual frame intervals
- **Highlight Protection Filter**: Luminance-squared mask fusion prevents over-exposure in bright regions after enhancement (disabled during real-time streaming for maximum throughput)
- **Cross-Platform GPU Acceleration**: Auto-detects CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
- **History Management**: Per-user processing history with admin/guest role separation via JSON persistence
- **Dual Interface**: Web GUI (Flask + Bootstrap 5) and CLI batch processing (`main.py`)

***

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Web Frontend (HTML5)                          │
│               Image/Video Upload  │  Real-time WebSocket Stream      │
└─────────────────────┬─────────────┴──────────────┬───────────────────┘
                      │                            │
┌─────────────────────▼────────────────────────────▼───────────────────┐
│                     Flask + Flask-SocketIO Server (app.py)           │
│                                                                      │
│   ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│   │ Upload Route  │   │ WebSocket Handler│   │ History Manager  │    │
│   │ (/upload)     │   │ (stream_frame)   │   │ (results.json)   │    │
│   └──────┬────────┘   └────────┬─────────┘   └──────────────────┘    │
│          │                     │                                     │
│   ┌──────▼─────────────────────▼───────────┐                         │
│   │       4-Thread Parallel Pipeline       │                         │
│   │  Thread A (Read) ──► Thread B (Enhance)│                         │
│   │       │                    │           │                         │
│   │       └────────────────────┤           │                         │
│   │                            ▼           │                         │
│   │  Thread C (Detect) ◄─── frame ◄────────│                         │
│   │       │                    │           │                         │
│   │       └────────┬───────────┘           │                         │
│   │                ▼                       │                         │
│   │  Thread D (Assemble + Tracking)        │                         │
│   └────────────────────────────────────────┘                         │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────────┐
│                      AI Inference Engine                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │   DarkIR     │  │  Zero-DCE    │  │  YOLOv8 + ByteTrack          │ │
│  │  (U-Net +    │  │ (Depthwise   │  │  (Detection + Multi-Object   │ │
│  │  FreqMLP)    │  │  Separable   │  │   Tracking + Counting)       │ │
│  │              │  │  CNN)        │  │                              │ │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

***

## Project Structure

```
.
├── app.py                          # Flask + SocketIO web server (main entry)
├── main.py                         # CLI tool for batch enhancement (5 modes)
├── model.py                        # Zero-DCE network definition (enhance_net_nopool)
├── image_filters.py                # Highlight protection post-processing filter
├── inference_DarkIR.py             # DarkIR inference API (image/video/folder)
├── inference_Zero_DCE.py           # Zero-DCE inference class (image/video/camera)
├── yolo_count_bytetrack_stable.py  # Stable ByteTrack tracking & counting logic
├── yolo_count_from_labels.py       # Counting utilities (zone bounds, CCTV header, count panel)
├── yolov8_predict.py               # YOLOv8 predictor (memory-level API + ByteTrack session)
├── requirement.txt                 # Python dependencies (exact versions)
├── results.json                    # Processing history records (auto-generated)
├── start_ngrok.bat                 # Ngrok tunnel startup script (optional)
├── openh264-1.8.0-win64.dll        # H.264 encoder library (Cisco OpenH264)
├── 启动说明.txt                     # Quick-start guide (Chinese)
│
├── archs/                          # DarkIR model architecture
│   ├── __init__.py                 # Model factory (create_model from YAML config)
│   ├── DarkIR.py                   # Main U-Net with PixelShuffle upsample + side loss
│   ├── arch_model.py               # EBlock, DBlock, FreMLP, SimpleGate, Branch
│   └── arch_util.py                # LayerNorm2d, CustomSequential
│
├── options/                        # DarkIR configuration
│   ├── DarkIR.yml                  # Network hyperparameters (channels, blocks, dilations)
│   └── options.py                  # YAML config parser with OrderedDict support
│
├── models/                         # Pretrained weights (gitignored, must download)
│   ├── best.pt                     # YOLOv8 pedestrian detection model
│   ├── DarkIR.pt                   # DarkIR enhancement weights
│   └── Zero_DCE.pth                # Zero-DCE enhancement weights
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Main page (upload + live stream + sidebar)
│   ├── result.html                 # Processing result view (image/video player)
│   └── history.html                # History records page (admin/guest filtered)
│
└── static/                         # Runtime data (gitignored, auto-created)
    ├── uploads/                    # User uploaded original files
    ├── results/                    # Processed output files (prefixed with "res_")
    └── live_records/               # Live stream recordings (prefixed with "live_")
```

***

## Dependencies

All Python dependencies with pinned versions are listed in [`requirement.txt`](requirement.txt).

| Category                  | Package         | Version      | Purpose                           |
| ------------------------- | --------------- | ------------ | --------------------------------- |
| **Web Framework**         | Flask           | 3.1.3        | HTTP routing & templating         |
| <br />                    | Flask-SocketIO  | 5.6.1        | WebSocket support                 |
| <br />                    | python-socketio | 5.16.2       | SocketIO protocol                 |
| <br />                    | python-engineio | 4.13.2       | Engine.IO transport               |
| **PyTorch (CUDA 12.1)**   | torch           | 2.5.1+cu121  | Deep learning framework           |
| <br />                    | torchvision     | 0.20.1+cu121 | Image transforms & models         |
| <br />                    | torchaudio      | 2.5.1+cu121  | Audio processing                  |
| **Computer Vision**       | opencv-python   | 4.10.0.84    | Image/video I/O & processing      |
| <br />                    | numpy           | 2.0.0        | Numerical computing               |
| <br />                    | Pillow          | 10.3.0       | Image handling                    |
| <br />                    | scikit-image    | 0.24.0       | Image analysis                    |
| <br />                    | imageio         | 2.37.3       | Multi-format image I/O            |
| <br />                    | kornia          | 0.7.2        | Differentiable CV ops             |
| <br />                    | kornia\_rs      | 0.1.10       | Kornia Rust backend               |
| **YOLO / Tracking**       | ultralytics     | 8.4.67       | YOLOv8 detection                  |
| <br />                    | filterpy        | 1.4.5        | Kalman filters (ByteTrack dep)    |
| <br />                    | lap             | 0.5.13       | Linear assignment (ByteTrack dep) |
| **Low-Light Enhancement** | einops          | 0.8.0        | Tensor operations                 |
| <br />                    | ptflops         | 0.7.3        | FLOPs calculation                 |
| **Utilities**             | PyYAML          | 6.0.1        | YAML config parsing               |
| <br />                    | tqdm            | 4.66.4       | Progress bars                     |
| <br />                    | matplotlib      | 3.10.8       | Plotting                          |
| <br />                    | scipy           | 1.13.1       | Scientific computing              |

**One-command install:**

```bash
pip install -r requirement.txt
```

> **Note**: The `torch` and `torchvision` versions above target CUDA 12.1. If you need CPU-only or a different CUDA version, install PyTorch manually from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirement.txt`.

***

## Installation

### Prerequisites

| Requirement                | Notes                                         |
| -------------------------- | --------------------------------------------- |
| **Python 3.10+**           | Recommended: 3.10.13                          |
| **NVIDIA GPU + CUDA 12.1** | Optional — auto-fallback to Apple MPS or CPU  |
| **Git**                    | For cloning the repository                    |
| **FFmpeg** (macOS)         | `brew install ffmpeg` — needed for mp4v codec |

### Download Model Weights

网盘链接: <https://pan.baidu.com/s/18x-qZ23T7ltvAKTBiFkdHA?pwd=khsq> 提取码: khsq

Before running the system, download the required pretrained model weights and place them in the `models/` directory:

```
models/
├── best.pt          # YOLOv8 pedestrian detection model
├── DarkIR.pt        # DarkIR enhancement weights
└── Zero_DCE.pth     # Zero-DCE enhancement weights
```

### Setup (Windows)

```powershell
# 1. Clone the repository
git clone <repo-url>
cd "Pedestrian Detection and Counting System in Low-Light Environments"

# 2. Install Python dependencies
pip install -r requirement.txt

# 3. (Optional) Install CUDA Toolkit 12.1 for GPU acceleration
#    Download: https://developer.nvidia.com/cuda-downloads

# 4. Start the web server
python app.py
#    → Open http://localhost:5000 in your browser
```

> **Admin password**: `666666` | **Default port**: `5000` (change at bottom of `app.py`)

### Setup (macOS)

```bash
# 1. Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install pyenv + xz (Python version management)
brew install pyenv xz

# 3. Install Python 3.10.13
pyenv install 3.10.13
pyenv local 3.10.13

# 4. Install dependencies
pip install flask flask-socketio numpy opencv-python Pillow torch torchvision ptflops tqdm ultralytics

# 5. (Optional) Install FFmpeg for mp4v codec support
brew install ffmpeg

# 6. Start the web server
python app.py
#    → Open http://localhost:5000 in your browser
```

> **Apple Silicon (M1/M2/M3)**: MPS acceleration is auto-detected and enabled. No additional configuration needed.

### CUDA Path Configuration

If CUDA is installed at a non-default location, update `cuda_bin_path` at the top of `app.py`:

```python
cuda_bin_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin'
```

On startup, `app.py` prints a full GPU diagnostic to verify acceleration is working correctly.

***

## Usage

### Web Application

1. Start the server: `python app.py`
2. Open `http://localhost:5000` in your browser
3. **Upload Processing**: Select image/video, configure enhance/detect/counting mode, click upload. View results with video player and counting stats.
4. **Real-time Streaming**: Click "Live Stream" tab, grant camera permission. Toggle enhance/detect, select counting zone, and optionally record the session.
5. **History**: View past processing results. Admins see all records; guests see only their own.

#### Supported Counting Configurations

| Mode            | Description                             | Web UI Parameter                               |
| --------------- | --------------------------------------- | ---------------------------------------------- |
| Full Image      | Line crossing count (entire frame)      | `zone.id = null, count_mode = 'full'`          |
| Zone (2×3 Grid) | 6-zone directional foot-region counting | `zone.id = 1-6, direction = top_to_bottom/...` |
| Global Line     | H1 / V1 / V2 directional line crossing  | `zone.line_id = h1/v1/v2, count_mode = 'line'` |

### Command Line Interface

`main.py` provides 5 operating modes for batch image/video enhancement:

```bash
# Single image enhancement (DarkIR)
python main.py --mode image --input inputs/test.jpg --output results/enhanced.jpg

# Batch image folder enhancement (DarkIR)
python main.py --mode batch --input inputs/ --output results/

# Single video enhancement
python main.py --mode video --input videos/test.mp4 --output results/enhanced.mp4 --model darkir
python main.py --mode video --input videos/test.mp4 --output results/enhanced.mp4 --model zero_dce

# Batch video folder enhancement
python main.py --mode video_batch --input videos/ --output results/ --model darkir
python main.py --mode video_batch --input videos/ --output results/ --model zero_dce

# Camera real-time enhancement (Zero-DCE)
python main.py --mode camera --camera_id 0
python main.py --mode camera --camera_id 0 --save_video --output camera.mp4
```

**CLI Arguments**:

| Argument       | Type | Default                | Description                               |
| -------------- | ---- | ---------------------- | ----------------------------------------- |
| `--mode`       | str  | (required)             | image, batch, video, video\_batch, camera |
| `--input`      | str  | —                      | Input file or directory path              |
| `--output`     | str  | `./results/`           | Output file or directory path             |
| `--model`      | str  | `darkir`               | darkir / zero\_dce (video modes only)     |
| `--config`     | str  | `./options/DarkIR.yml` | DarkIR config file path                   |
| `--camera_id`  | int  | `0`                    | Camera device ID                          |
| `--save_video` | flag | False                  | Save camera output to file                |
| `--no_display` | flag | False                  | Disable preview window                    |

***

## Configuration Guide

### Model File Paths

Ensure the following paths in `app.py` are correct:

```python
detector = YoloMemoryPredictor('models/best.pt')                     # Line ~107
DARKIR_MODEL, darkir_resize = darkir_api.init_model('./options/DarkIR.yml')  # Line ~111
ZERO_DCE_ENGINE = ZeroDCEInference(model_path='models/Zero_DCE.pth', ...)     # Line ~114
```

### Key Tuning Parameters

| Parameter          | Location                         | Default    | Description                                             |
| ------------------ | -------------------------------- | ---------- | ------------------------------------------------------- |
| `scale_factor`     | `inference_Zero_DCE.py`          | 12         | Zero-DCE downsampling factor (larger = faster, coarser) |
| `conf` / `iou`     | `yolov8_predict.py`              | 0.2 / 0.45 | YOLO detection thresholds                               |
| `track_buffer`     | `yolo_count_bytetrack_stable.py` | 60         | ByteTrack lost-track retention frames                   |
| `match_thresh`     | `yolo_count_bytetrack_stable.py` | 0.8        | ByteTrack IoU matching threshold                        |
| `hysteresis_ratio` | `yolo_count_bytetrack_stable.py` | 0.035      | Counting line hysteresis zone size                      |
| `warmup_frames`    | `yolo_count_bytetrack_stable.py` | 5          | Initial frames before IN/OUT counting                   |
| `port`             | `app.py` (last line)             | 5000       | Web server port                                         |

### Ngrok Tunnel (Optional)

For public access, use the bundled `start_ngrok.bat` script with [ngrok](https://ngrok.com/).

***

## Methodology

### 1. Low-Light Image Enhancement

#### DarkIR (Frequency-Domain U-Net)

DarkIR is a symmetric U-Net architecture designed for low-light image enhancement, incorporating frequency-domain processing and multi-scale dilated convolutions for superior image quality.

**Network Config** (`options/DarkIR.yml`):

| Parameter                | Value      | Description                                       |
| ------------------------ | ---------- | ------------------------------------------------- |
| `img_channels`           | 3          | Input RGB channels                                |
| `width`                  | 64         | Base feature channels (doubled per encoder layer) |
| `enc_blk_nums`           | \[1, 2, 3] | EBlock counts per encoder stage                   |
| `dec_blk_nums`           | \[3, 1, 1] | DBlock counts per decoder stage                   |
| `middle_blk_num_enc/dec` | 2          | Bottleneck block counts                           |
| `dilations`              | \[1, 4, 9] | Multi-scale dilation rates                        |
| `extra_depth_wise`       | True       | Extra depthwise conv in EBlock/DBlock             |

**Encoder Block (EBlock)**:

- **Multi-scale dilated convolution branches**: Parallel depthwise convolutions with dilation rates \[1, 4, 9] capture features at different receptive fields
- **Channel Self-Attention (SCA)**: Global average pooling + 1×1 convolution produces channel-wise attention weights
- **SimpleGate**: Splits feature channels into halves and multiplies element-wise: `G(x₁, x₂) = x₁ · x₂`
- **Frequency-domain MLP (FreMLP)**: Applies real FFT → processes only the magnitude spectrum through a 2-layer MLP while preserving phase → inverse FFT. Captures global context efficiently without quadratic complexity

**Decoder Block (DBlock)**:

- Same multi-scale dilated convolution + SCA + SimpleGate as EBlock
- Uses a standard Feed-Forward Network (FFN) instead of FreMLP
- Both EBlock and DBlock employ learnable residual scaling parameters (β, γ) initialized to zero

**Overall Architecture**:

- Symmetric U-Net with PixelShuffle for upsampling (channel-to-spatial conversion)
- Skip connections between encoder and decoder stages
- Global residual connection: `output = input + ending(decoder_features)`
- Auto-padding to ensure divisibility by 8 at each downsample stage
- Optional large-image strategy: downsample by 2× before enhance, upsample result back

#### Zero-DCE (Zero-Reference Deep Curve Estimation)

Zero-DCE is a lightweight CNN that estimates pixel-wise enhancement curve parameters without paired training data. The model is defined in [model.py](model.py).

- **Depthwise Separable Convolutions (CSDN\_Tem)**: Each block = depthwise conv (groups=in\_ch) + pointwise conv (1×1), drastically reducing parameters
- **7-layer encoder with skip connections**: Concatenates early and later layer features for multi-scale representation
- **Iterative Higher-Order Curve**: Applies the quadratic curve 8 times iteratively:
  ```
  LE_n(x) = LE_{n-1}(x) + A_n(x) · (LE_{n-1}(x)² - LE_{n-1}(x))
  ```
  where `A_n(x)` is the pixel-wise curve parameter map (bounded by tanh to \[-1, 1])
- **Scalable inference**: Input downsampled by `scale_factor` (default 12), then curve map upsampled back. Larger factor = faster but coarser

| Model    | FLOPs (3×256×256) | Advantages                                  | Use Case                         |
| -------- | ----------------- | ------------------------------------------- | -------------------------------- |
| DarkIR   | \~15-20 GMACs     | Superior quality, frequency-domain features | Image processing, offline video  |
| Zero-DCE | \~4-5 GMACs       | Fast, lightweight, real-time capable        | Live stream, WebSocket streaming |

### 2. Pedestrian Detection (YOLOv8)

Uses Ultralytics YOLOv8 with a custom-trained pedestrian detection model (`models/best.pt`):

- **Device auto-selection**: CUDA → MPS → CPU
- **Memory-level API** (`YoloMemoryPredictor` in [yolov8\_predict.py](yolov8_predict.py)): Accepts numpy frames directly, returns detection boxes — zero disk I/O
- **Detection parameters**: conf=0.2, iou=0.45, classes=\[0] (person only)
- **ByteTrackSession**: Per-client tracker instances in `YoloMemoryPredictor` for WebSocket streaming

### 3. Multi-Object Tracking (ByteTrack)

Two separate ByteTrack tracker implementations serve different purposes:

1. **`yolov8_predict.py::ByteTrackSession`** — Lightweight tracker for basic per-client tracking in WebSocket streaming
2. **`yolo_count_bytetrack_stable.py::StableByteTrackSession`** — Specialized tracker tuned for low-light counting with:
   - Higher `track_buffer=60` for occlusion resilience
   - Fuse score enabled for better track quality
   - Stable ID reassignment via center+foot distance matching (combats ByteTrack ID switches)

**Stable ID Reconnection** ([`_assign_stable_ids`](yolo_count_bytetrack_stable.py)): When ByteTrack assigns a new raw ID, the system searches recent known tracks within a distance gate (weighted 70% foot + 30% center distance). This prevents counting errors from ID fragmentation.

### 4. Three Counting Modes

All modes use ByteTrack tracking with hysteresis bands to avoid boundary jitter.

#### Full-Image Line Crossing

- Horizontal line at frame center (top\_to\_bottom by default)
- Each person counted once on first directional crossing
- Configurable hysteresis margin

#### Zone-Based Counting (2×3 Grid)

- Frame divided into 6 zones (2 rows × 3 cols)
- Counting line placed at `line_percent%` inside the selected zone boundary
- Directional: top\_to\_bottom, bottom\_to\_top, left\_to\_right, right\_to\_left
- Foot-region mode (`process_frame_memory_zones_foot_region`): Tracks bottom-center point trajectories for more accurate enter/exit detection with confirm frames and duplicate suppression

#### Global Line Counting

- Three pre-defined counting lines: H1 (horizontal center), V1 (left third), V2 (right third)
- Full-frame ByteTrack tracking with foot-point trajectory visualization
- Directional counting with hysteresis

### 5. Post-Processing: Highlight Protection

The [highlight protection filter](image_filters.py) prevents over-exposure in bright regions (e.g., lamps, windows) after low-light enhancement:

1. Compute luminance mask from original frame: `mask = (gray / 255)²`
2. Weighted fusion: `output = original × mask + enhanced × (1 - mask)`
3. Bright areas retain more original pixels; dark areas use enhanced result

> **Note**: This filter is disabled during real-time WebSocket streaming to maximize throughput.

***

## System Pipeline

### Video Processing (4-Thread Parallel Pipeline)

```
┌────────────────────────────────────────────────────────────┐
│              process_video_file() in app.py                │
├────────────────────────────────────────────────────────────┤
│  Thread A: Read & Decimate                                 │
│  ├─ Read frames from video file                            │
│  ├─ Decimate to ≤24 FPS (sample_interval = orig_fps / 24)  │
│  ├─ Resize to ≤1080p (short edge limit)                    │
│  └─ Feed both q_raw_dark and q_raw_yolo queues             │
│                                                            │
│  Thread B: Enhance (parallel with Thread C)                │
│  ├─ DarkIR or Zero-DCE based on enhance_model param        │
│  ├─ Apply highlight protection filter                      │
│  └─ Store in Image_Buffer_Dict                             │
│                                                            │
│  Thread C: YOLO Detect (parallel with Thread B)            │
│  ├─ Run detector.predict_frame()                           │
│  └─ Store in BBox_Buffer_Dict                              │
│                                                            │
│  Thread D: Assemble & Write                                │
│  ├─ Poll for both buffers ready                            │
│  ├─ ByteTrack tracking + counting overlay                  │
│  ├─ Write to output MP4 (avc1 codec)                       │
│  └─ Update processing_progress for frontend polling        │
└────────────────────────────────────────────────────────────┘
```

### Real-Time WebSocket Streaming

```
Frontend (Webcam) ──WebSocket──► handle_stream_frame()
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               │               ▼
            task_enhance()          │        task_detect()
            (Zero-DCE GPU)          │        (YOLOv8)
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                          Assemble + ByteTrack
                                    │
                                    ▼
                          JPEG encode → emit('stream_result')
```

- Frame-skip debounce: drops incoming frames if previous frame is still processing
- `ThreadPoolExecutor` (max\_workers=4) for concurrent enhance + detect
- Zero-DCE runs entirely on GPU (tensor ops) — no CPU numpy bottleneck
- Recording: buffers first 5 frames to compute real FPS, then writes to MP4

***

## API Reference

### REST Endpoints

| Method | Route                  | Auth     | Description                            |
| ------ | ---------------------- | -------- | -------------------------------------- |
| `GET`  | `/`                    | —        | Main page (assigns session UID)        |
| `POST` | `/login`               | password | Admin login (`{"password": "666666"}`) |
| `GET`  | `/logout`              | session  | Clear admin session                    |
| `POST` | `/upload`              | —        | Upload image/video for processing      |
| `GET`  | `/view/<filename>`     | —        | View processing result                 |
| `GET`  | `/history`             | —        | Processing history (admin sees all)    |
| `POST` | `/delete_history`      | session  | Delete a history record                |
| `GET`  | `/progress/<filename>` | —        | Poll video processing progress         |

#### Upload Request Parameters

| Field           | Type | Default      | Description                                                                                                     |
| --------------- | ---- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| `file`          | file | (required)   | Image or video file                                                                                             |
| `enhance`       | str  | `"true"`     | Enable low-light enhancement                                                                                    |
| `detect`        | str  | `"true"`     | Enable pedestrian detection                                                                                     |
| `enhance_model` | str  | `"zero_dce"` | Enhancement model for video (darkir / zero\_dce)                                                                |
| `zone`          | JSON | `null`       | Counting config: `{"id": 1, "mode": "zone", "direction": "top_to_bottom", "line_percent": 12, "line_id": "h1"}` |

### WebSocket Events

| Event           | Direction       | Description                                |
| --------------- | --------------- | ------------------------------------------ |
| `stream_frame`  | Client → Server | Send video frame as base64 JPEG + params   |
| `stream_result` | Server → Client | Receive processed frame + count + zone\_id |
| `disconnect`    | Server          | Auto-save recording, clean up client state |

#### `stream_frame` Payload

```json
{
  "image": "data:image/jpeg;base64,...",
  "enhance": true,
  "detect": true,
  "quality": "480p",
  "record": false,
  "zone": {
    "id": 1,
    "mode": "zone",
    "direction": "top_to_bottom",
    "line_percent": 12,
    "line_id": "h1"
  }
}
```

#### `stream_result` Response

```json
{
  "status": "success",
  "image": "data:image/jpeg;base64,...",
  "count": "CUR 3 / IN 5 / OUT 2",
  "zone_id": 1
}
```

***

## Performance Optimization

### Video Processing

- **Auto 1080p limit**: Inputs >1080p are downscaled (short edge capped at 1080)
- **24 FPS cap**: High-FPS videos are decimated by `sample_interval = original_fps / 24`
- **4-thread pipeline**: Read, Enhance, Detect run concurrently; Assemble serializes after both complete
- **avc1 codec**: Hardware-accelerated H.264 encoding on supported GPUs

### WebSocket Streaming

- **GPU-only Zero-DCE**: Image preprocessing (`HWC→CHW`, `/255.0`, BGR→RGB) runs entirely as tensor ops on GPU
- **ThreadPoolExecutor**: Enhance and detect run concurrently per frame
- **Frame-skip debounce**: Drops incoming frames when previous is still processing
- **No highlight protection**: Disabled during streaming (it was the primary FPS bottleneck)
- **JPEG quality**: 85 for 720p, 70 for 480p

### GPU Diagnostics

On startup, `app.py` prints a full GPU diagnostic:

```
==================================================
  GPU / 加速诊断
==================================================
  PyTorch 版本: 2.5.1+cu121
  CUDA 可用: True
  CUDA 版本: 12.1
  GPU 型号: NVIDIA GeForce RTX 3060
  GPU 数量: 1
  当前设备: cuda:0
  CUDA 测试: 张量运算正常
  DarkIR 设备: cuda
==================================================
```

***

## Troubleshooting

| Issue                                    | Solution                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `FileNotFoundError: models/best.pt`      | Download model weights to `models/` directory                                         |
| CUDA not detected                        | Verify CUDA Toolkit 12.1 is installed; check `cuda_bin_path` in `app.py`              |
| `ImportError: DLL load failed` (Windows) | Install [Visual C++ Redistributables](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| WebSocket connection drops               | Increase `ping_timeout` in Flask-SocketIO config; check firewall                      |
| Low FPS on CPU                           | Use `--model zero_dce` or `enhance_model='zero_dce'` — Zero-DCE is much lighter       |
| `CUDA out of memory`                     | Reduce input resolution or switch to `zero_dce` for video                             |
| mp4v codec not available (macOS)         | Install FFmpeg: `brew install ffmpeg`                                                 |

***

## License

This project is created as a university course assignment (专业综合设计). Contact <lqboyh@gmail.com> for usage inquiries.

***

#   P e d e s t r i a n - D e t e c t i o n - a n d - C o u n t i n g - S y s t e m - i n - L o w - L i g h t - E n v i r o n m e n t s  
 