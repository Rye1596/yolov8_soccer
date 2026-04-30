# 足球比赛智能分析系统说明书

## 1. 系统概述 (System Overview)

本系统是一个基于深度学习和计算机视觉技术的足球比赛视频智能分析平台。它能够自动检测视频中的球员、裁判和足球，进行持续跟踪，并提供高级数据分析功能，如球员速度计算、跑动距离统计、控球率分析以及球场平面视角的映射。系统集成了现代化的 Web 界面，方便用户上传视频、选择模型并查看分析结果。

## 2. 系统架构 (System Architecture)

系统采用 B/S（浏览器/服务器）架构，主要包含以下三个部分：

### 2.1 前端 (Frontend)
- **技术栈**: HTML5, CSS3, JavaScript (原生)
- **功能**:
    - 视频上传与管理
    - YOLO 模型选择
    - 实时分析进度显示
    - 历史记录查看
    - AI 文件分析助手（基于智谱 AI）

### 2.2 后端 (Backend)
- **技术栈**: Python, Flask
- **功能**:
    - 提供 RESTful API 接口
    - 处理视频文件上传与存储
    - 调度核心分析算法
    - 管理分析任务与结果

### 2.3 核心算法层 (Core Analysis)
- **目标检测**: 集成 YOLOv8, YOLOv11, YOLOv12 等先进模型。
- **目标跟踪**: 使用 ByteTrack/Supervision 进行多目标跟踪。
- **数据分析**: 包含多个专用模块（详见下文）。

## 3. 功能模块详解 (Functional Modules)

### 3.1 目标检测与跟踪 (Object Detection & Tracking)
- **模块**: `trackers/`
- **功能**: 识别视频中的球员、裁判和足球，并为每个目标分配唯一的 ID，实现跨帧跟踪。支持处理遮挡和目标重识别。

### 3.2 队伍分配 (Team Assignment)
- **模块**: `team_assigner/`
- **算法**: 基于 K-Means 聚类算法。
- **功能**: 分析球员球衣颜色，自动将球员分配到两支不同的队伍，并区分裁判。

### 3.3 控球判定 (Ball Possession)
- **模块**: `player_ball_assigner/`
- **功能**: 计算足球与球员的距离，判定当前控球球员，进而统计各队的控球时间比例。

### 3.4 相机运动估计 (Camera Movement Estimation)
- **模块**: `camera_movement_estimator/`
- **算法**: 光流法 (Optical Flow)。
- **功能**: 计算相机在每一帧的平移量，用于修正目标在世界坐标系中的位置，消除相机抖动或运镜带来的误差。

### 3.5 视角变换 (View Transformation)
- **模块**: `view_transformer/`
- **算法**: 单应性矩阵变换 (Homography)。
- **功能**: 将视频像素坐标映射到标准的 2D 足球场平面坐标，实现球员位置的战术视图还原。

### 3.6 速度与距离计算 (Speed & Distance Estimation)
- **模块**: `speed_and_distance_estimate/`
- **功能**: 基于变换后的真实世界坐标，计算球员的瞬时速度和累计跑动距离。

## 4. 目录结构说明 (Directory Structure)

```
project/
├── app.py                      # Flask Web 后端入口
├── main.py                     # 核心分析流程入口
├── requirements.txt            # 项目依赖文件
├── web/                        # 前端静态资源
│   ├── index.html              # 主页
│   ├── css/                    # 样式文件
│   └── js/                     # 脚本文件
├── models/                     # YOLO 模型权重文件 (.pt)
├── trackers/                   # 跟踪器模块
├── team_assigner/              # 队伍分配模块
├── player_ball_assigner/       # 控球判定模块
├── camera_movement_estimator/  # 相机运动估计模块
├── view_transformer/           # 视角变换模块
├── speed_and_distance_estimate/# 速度距离计算模块
├── utils/                      # 通用工具函数
└── output_videos/              # 分析结果输出目录
```

## 5. 环境要求与安装 (Installation)

### 5.1 环境要求
- Python 3.8+
- CUDA 支持（推荐，用于 GPU 加速）

### 5.2 安装依赖
在项目根目录下运行：
```bash
pip install -r requirements.txt
```

### 5.3 启动系统
1. **启动 Web 服务器**:
   ```bash
   python app.py
   ```
   或者运行 `web/start_server.bat` (Windows)。

2. **访问界面**:
   打开浏览器访问 `http://localhost:5000` (默认端口)。

## 6. 使用说明 (Usage Guide)

1. **上传视频**: 在网页主界面点击上传区域，选择本地的足球比赛视频文件（支持 mp4, avi 等格式）。
2. **选择模型**: 在左侧控制面板选择合适的 YOLO 模型（如 `soccer_yolov8x.pt`）。
3. **开始分析**: 点击"开始分析"按钮，系统将后台处理视频。
4. **查看结果**: 分析完成后，页面将展示处理后的视频，包含目标框、轨迹、速度数据和控球统计。
5. **AI 助手**: 点击"AI 文件分析"可与智能助手对话，了解项目代码或分析数据细节。
