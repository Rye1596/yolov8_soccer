# 足球比赛分析系统 - Streamlit部署指南

## 📋 项目概述

本项目是一个基于YOLOv8的足球比赛智能分析系统，提供以下功能：
- 球员、裁判、足球目标检测与跟踪
- 自动队伍识别（基于球衣颜色）
- 球权分析与控球率统计
- 摄像机运动补偿
- 球员速度与距离估算
- 可视化结果输出

## 🚀 本地部署步骤

### 1. 环境准备

确保已安装Python 3.8或更高版本：
```bash
python --version
```

### 2. 克隆/下载项目

```bash
cd c:\Users\Rye\Desktop\yolo\football_tracking
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备模型文件

确保`models/`目录中包含YOLO模型文件（.pt格式）：
- soccer_yolov8x.pt
- soccer_yolov8x2.pt
- soccer_yolov11x.pt
- soccer_yolov12x.pt

### 5. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址：`http://localhost:8501`

## 🌐 Streamlit Cloud部署

### 1. 准备GitHub仓库

将项目推送到GitHub仓库：
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/football-tracking.git
git push -u origin main
```

### 2. 创建配置文件

创建`.streamlit/config.toml`：
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### 3. 创建packages.txt（如果需要系统依赖）

```
libgl1-mesa-glx
libglib2.0-0
```

### 4. 部署到Streamlit Cloud

1. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
2. 使用GitHub账号登录
3. 点击"New app"
4. 选择你的仓库和分支
5. 指定主文件路径：`app.py`
6. 点击"Deploy"

## ⚙️ 配置说明

### 模型配置

在侧边栏可以选择不同的YOLO模型：
- **yolov8n**: 最快但精度较低
- **yolov8s**: 平衡速度和精度
- **yolov8m**: 中等精度
- **yolov8l**: 高精度
- **yolov8x**: 最高精度但最慢

### 视频要求

- 格式：MP4、AVI、MOV、MKV
- 建议分辨率：720p或1080p
- 建议时长：30秒-5分钟（处理时间与视频长度成正比）

## 🔧 常见问题

### 1. CUDA/GPU支持

如果需要使用GPU加速，确保安装CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存不足

对于长视频，可以：
- 降低视频分辨率
- 减少处理帧数
- 使用更小的模型

### 3. 模型文件缺失

如果提示模型文件不存在，请确保：
- 模型文件放在`models/`目录下
- 文件名正确（区分大小写）
- 文件格式为.pt

## 📁 项目结构

```
football_tracking/
├── app.py                      # Streamlit主应用
├── requirements.txt            # Python依赖
├── DEPLOY.md                   # 部署文档
├── models/                     # YOLO模型文件
│   ├── soccer_yolov8x.pt
│   └── ...
├── input_videos/               # 输入视频目录
├── output_videos/              # 输出视频目录
├── trackers/                   # 跟踪器模块
│   └── tracker.py
├── team_assigner/              # 队伍分配模块
│   └── team_assigner.py
├── player_ball_assigner/       # 球权分配模块
│   └── play_ball_assigner.py
├── camear_movement_estimator/  # 摄像机运动估计
│   └── camera_movement_estimator.py
├── view_transformer/           # 视角变换
│   └── view_transformer.py
├── speed_and_distance_estimate/# 速度距离估算
│   └── speed_and_distance_estimate.py
└── utils/                      # 工具函数
    ├── video_utils.py
    └── bbox_utils.py
```

## 📝 使用说明

1. **上传视频**：在左侧边栏上传足球比赛视频
2. **选择模型**：从下拉菜单选择合适的YOLO模型
3. **开始分析**：点击"开始分析"按钮
4. **查看结果**：
   - 左侧显示原始视频
   - 右侧显示分析结果
   - 底部显示统计信息
5. **下载结果**：点击下载按钮保存分析后的视频

## 🔒 注意事项

- 首次运行需要下载YOLO模型，可能需要几分钟
- 视频处理时间较长，请耐心等待
- 建议使用较短的视频进行测试
- 确保有足够的磁盘空间存储临时文件

## 📞 技术支持

如有问题，请检查：
1. 所有依赖是否正确安装
2. 模型文件是否存在
3. 视频格式是否支持
4. 查看控制台错误信息

## 🔄 更新日志

### v1.0.0
- 初始版本发布
- 支持视频上传和分析
- 实现基本的检测和跟踪功能
- 添加队伍识别和球权分析
