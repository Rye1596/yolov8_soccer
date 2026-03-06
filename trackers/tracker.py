# 导入必要的库
from ultralytics import YOLO  # YOLO目标检测模型
import supervision as sv      # 计算机视觉监控工具库
import pickle                 # 用于保存和加载数据的序列化模块
import os                     # 文件系统操作
import sys                    # 系统参数和功能访问
import cv2                    # OpenCV库，用于图像处理
import numpy as np            # 数值计算库
import pandas as pd           # 数据处理库，用于球位置插值

# 添加项目根目录到路径，以便导入自定义模块
sys.path.append('../')
# 从自定义工具模块导入函数
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """
    目标跟踪器类，负责检测、跟踪和绘制球员、裁判和足球
    使用YOLO模型进行目标检测，ByteTrack进行多目标跟踪
    """
    
    def __init__(self, model_path):
        """
        初始化跟踪器
        
        参数:
            model_path: str - YOLO模型文件的路径
        """
        # 加载YOLO模型
        self.model = YOLO(model_path)
        # 初始化ByteTrack多目标跟踪器
        self.tracker = sv.ByteTrack()
        # 存储stub文件路径，用于保存/加载跟踪结果
        self.stub_path = None
    def add_position_to_tracks(self, tracks):
        """
        为跟踪结果添加位置信息
        
        参数:
            tracks: dict - 包含所有目标跟踪结果的字典
        
        处理逻辑:
            - 为球员和裁判添加脚部位置（边界框底部中点）
            - 为足球添加中心位置
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if object_type == "players" or object_type == "referees":
                    for track_id, track_info in track.items():
                        bbox = track_info['bbox']
                        # 获取球员/裁判脚部位置
                        position = get_foot_position(bbox)
                        tracks[object_type][frame_num][track_id]['position'] = position
                if object_type == "ball" and track.get("bbox") is not None:
                    bbox = track.get('bbox')
                    # 获取足球中心位置
                    position = get_center_of_bbox(bbox)
                    tracks[object_type][frame_num]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        对球的位置进行插值处理，填补检测缺失的帧
        
        参数:
            ball_positions: list - 包含每帧球位置的列表
            
        返回:
            list - 插值后的球位置列表
        """
        # 提取所有球的边界框
        ball_positions = [x.get('bbox', []) for x in ball_positions]
        # 将球位置转换为DataFrame格式
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # 使用插值方法填补缺失值
        df_ball_positions = df_ball_positions.interpolate()  # 线性插值填充大部分缺失值
        df_ball_positions = df_ball_positions.bfill()        # 向后填充，处理开头几帧没有值的情况

        # 将DataFrame转换回原始格式
        ball_positions = [{"bbox": x} for x in df_ball_positions.to_numpy().tolist()]

        # 以下代码为保存插值后的球位置到stub文件的功能（当前被注释）
        # if self.stub_path is not None:
        #     try:
        #         # 先读取原始数据
        #         with open(self.stub_path, 'rb') as f:
        #             all_tracks = pickle.load(f)
                
        #         # 更新球的位置数据
        #         all_tracks["ball"] = ball_positions
                
        #         # 保存回文件
        #         with open(self.stub_path, 'wb') as f:
        #             pickle.dump(all_tracks, f)
        #         print(f"插值后的球位置数据已保存至: {self.stub_path}")
        #     except Exception as e:
        #         print(f"保存插值数据失败: {e}")

        return ball_positions

    def detect_frames(self, frames):
        """
        使用YOLO模型批量检测视频帧中的目标
        
        参数:
            frames: list - 视频帧列表
            
        返回:
            list - 每帧的检测结果
        """
        # 定义批量处理大小（每次处理20帧）
        batch_size = 20
        detections = []
        
        # 批量处理视频帧
        for i in range(0, len(frames), batch_size):
            # 对当前批次的帧进行检测，置信度阈值设为0.1
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            
            # 注：使用predict而非track是因为守门员数据较少，不稳定，在守门员和球员之间跳来跳去
            # 后续会将守门员转换为球员类别
            
            # 将当前批次的检测结果添加到总结果中
            detections += detections_batch
            
            # 调试用代码（当前被注释）
            # break
            
        return detections
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        获取视频帧中所有目标的跟踪结果
        
        参数:
            frames: list - 视频帧列表
            read_from_stub: bool - 是否从stub文件读取跟踪结果
            stub_path: str - stub文件路径
            
        返回:
            dict - 包含球员、裁判和足球跟踪结果的字典
        """
        # 存储stub文件路径
        self.stub_path = stub_path
        
        # 如果选择从stub文件读取且文件存在，则直接返回跟踪结果
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        # 否则，使用YOLO模型检测目标
        detections = self.detect_frames(frames)
        
        # 初始化跟踪结果字典
        tracks = {
            "players": [],  # 球员跟踪结果
            "referees": [], # 裁判跟踪结果
            "ball": []      # 足球跟踪结果
        }
        
        # 处理每帧的检测结果
        for frame_num, detection in enumerate(detections):
            # 获取类别名称和类别索引映射
            cls_names = detection.names
            cls_names_idx = {v: k for k, v in cls_names.items()}
            
            # 将YOLO检测结果转换为supervision格式
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # 将守门员转换为球员类别
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_idx["player"]
            
            # 使用ByteTrack更新跟踪结果
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # 为当前帧初始化跟踪结果
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # 记录球员和裁判的跟踪结果
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # 边界框坐标
                cls_id = frame_detection[3]          # 类别ID
                track_id = frame_detection[4]        # 跟踪ID
            
                # 根据类别将结果添加到对应跟踪字典中
                if cls_id == cls_names_idx["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_idx["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
            # 记录足球的跟踪结果（足球不使用ByteTrack，直接使用YOLO检测结果）
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_idx["ball"]:
                    # print(f"frame_num:{frame_num}",frame_detection)  # 调试用代码
                    # break
                    tracks["ball"][frame_num] = {"bbox": bbox}
        
        # 如果提供了stub路径，则将跟踪结果保存到文件
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        return tracks
    def draw_ellipse(self, frame, bbox, color, track_id=None, thickness=2):
        """
        在帧中绘制表示球员/裁判位置的椭圆和跟踪ID
        
        参数:
            frame: numpy.ndarray - 视频帧
            bbox: list - 目标的边界框坐标 [x1, y1, x2, y2]
            color: tuple - 绘制颜色 (B, G, R)
            track_id: int - 目标的跟踪ID（可选）
            thickness: int - 绘制线条粗细（默认2）
            
        返回:
            numpy.ndarray - 绘制后的视频帧
        """
        # 获取边界框底部y坐标
        y2 = int(bbox[3])

        # 计算边界框中心点的x坐标和边界框宽度
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # 绘制椭圆（表示球员/裁判的位置）
        cv2.ellipse(frame,
                    center=(x_center, y2),                # 椭圆中心（球员脚部）
                    axes=(int(width/2), int(0.35*width)), # 椭圆轴长（基于边界框宽度）
                    angle=0.0,                             # 旋转角度
                    startAngle=-45,                        # 起始角度
                    endAngle=235,                          # 结束角度
                    color=color,                           # 绘制颜色
                    thickness=thickness,                   # 线条粗细
                    lineType=cv2.LINE_4)                   # 线条类型
                    
        # 绘制球员/裁判编号
        rectangle_width = 40    # 编号矩形宽度
        rectangle_height = 20   # 编号矩形高度
        
        # 计算编号矩形的坐标
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15  # 位置在椭圆下方
        y2_rect = (y2 + rectangle_height // 2) + 15
        
        if track_id is not None:
            # 绘制编号背景矩形
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
                          
            # 计算文字位置（根据ID位数调整）
            x1_text = x1_rect + 12
            if track_id > 99:  # 三位数ID需要左移一些
                x1_text -= 10
                
            # 绘制编号文字
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        color=(0, 0, 0),        # 文字颜色（黑色）
                        thickness=2,             # 文字粗细
                        fontScale=0.6           # 文字大小
                        ) 
        
        return frame
    def draw_traingle(self, frame, bbox, color):
        """
        在帧中绘制表示足球位置的三角形
        
        参数:
            frame: numpy.ndarray - 视频帧
            bbox: list - 足球的边界框坐标 [x1, y1, x2, y2]
            color: tuple - 绘制颜色 (B, G, R)
            
        返回:
            numpy.ndarray - 绘制后的视频帧
            
        注意: 方法名存在拼写错误，应为'draw_triangle'
        """
        # 获取边界框顶部y坐标
        y = int(bbox[1])
        # 计算边界框中心点的x坐标
        x, _ = get_center_of_bbox(bbox)

        # 定义三角形的三个顶点坐标（表示足球位置）
        triangle_points = np.array([
            [x, y],         # 底部中点
            [x-10, y-20],   # 左上方点
            [x+10, y-20],   # 右上方点
        ])
        
        # 绘制填充的三角形（表示足球）
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # 绘制三角形的黑色边框
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        
        return frame
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        在帧中绘制球队控球时间百分比信息
        
        参数:
            frame: numpy.ndarray - 视频帧
            frame_num: int - 当前帧编号
            team_ball_control: numpy.ndarray - 每帧控球球队的数组（1或2）
            
        返回:
            numpy.ndarray - 绘制后的视频帧
        """
        # 创建半透明矩形背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4  # 透明度
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 获取截至当前帧的控球数据
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # 计算各队控球帧数（使用numpy数组操作）
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # 计算控球百分比
        team_1_percentage = team_1_num_frames / (team_1_num_frames + team_2_num_frames) * 100
        team_2_percentage = team_2_num_frames / (team_1_num_frames + team_2_num_frames) * 100

        # 绘制球队1控球百分比
        cv2.putText(frame, 
                    f"Team 1 Ball Control: {team_1_percentage:.2f}%", 
                    (1400, 900), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 0), 
                    thickness=3)
        # 绘制球队2控球百分比
        cv2.putText(frame, 
                    f"Team 2 Ball Control: {team_2_percentage:.2f}%", 
                    (1400, 950), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 0), 
                    thickness=3)

        return frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        在视频帧中绘制所有注释信息
        
        参数:
            video_frames: list - 原始视频帧列表
            tracks: dict - 包含所有目标跟踪结果的字典
            team_ball_control: numpy.ndarray - 每帧控球球队的数组
            
        返回:
            list - 绘制了所有注释的视频帧列表
        """
        output_frames = []
        
        # 调试信息（当前被注释）
        # print("length of frame", len(video_frames))
        # print("length of tracks", len(tracks['players']))
        # print("shape of input video frames", video_frames[0].shape)
        
        # 处理每帧视频
        for frame_num, frame in enumerate(video_frames):
            # 创建帧的副本，避免修改原始帧
            frame = frame.copy()

            # 获取当前帧的跟踪数据
            player_dict = tracks['players'][frame_num]
            referees_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            
            # 绘制球员
            for track_id, player in player_dict.items():
                # 获取球员颜色（默认红色）
                color = player.get('color', (0, 0, 255))
                # 绘制球员椭圆和跟踪ID
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id, thickness=2)
                # 如果球员控球，在其位置绘制红色三角形
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player['bbox'], (0, 0, 255))
                    
            # 绘制裁判（使用黄色）
            for track_id, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id, thickness=2)
                
            # 绘制足球（使用绿色）
            if 'bbox' in ball_dict and len(ball_dict['bbox']) == 4:
                frame = self.draw_traingle(frame, ball_dict['bbox'], (0, 255, 0))

            # 绘制球队控球信息
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            # 将绘制后的帧添加到输出列表
            output_frames.append(frame)
            
        return output_frames
