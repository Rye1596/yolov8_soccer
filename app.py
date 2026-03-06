#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
足球比赛分析系统 - Streamlit Web应用
基于YOLOv8的目标检测与跟踪系统
"""
import streamlit as st
import cv2
import tempfile
import os
import sys
import numpy as np
import time
from pathlib import Path
import base64
from io import BytesIO
import shutil

# 设置页面配置
st.set_page_config(
    page_title="足球比赛智能分析系统",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.play_ball_assigner import PlayerBallAssigner
from camear_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimate.speed_and_distance_estimate import SpeedAndDistance_Estimate
from utils.video_utils import read_video
from utils.bbox_utils import get_center_of_bbox, measure_distance, get_foot_position
from data_analysis import FootballDataAnalyzer


def cleanup_temp_files():
    """清理临时文件"""
    temp_files = st.session_state.get('temp_files', [])
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
    st.session_state['temp_files'] = []


def save_video_to_file(frames, output_path, fps=12):
    """
    将帧保存为视频文件，使用浏览器兼容的编码格式
    
    参数:
        frames: 帧列表
        output_path: 输出文件路径
        fps: 帧率（默认12fps以减小文件大小）
    
    返回:
        str: 实际使用的文件路径，失败返回None
    """
    if not frames:
        return None
    
    height, width = frames[0].shape[:2]
    
    # 降低分辨率以减小文件大小（最大宽度1280）
    max_width = 1280
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_frames.append(resized_frame)
        frames = resized_frames
        height, width = new_height, new_width
    
    # 尝试不同的编码器，优先使用浏览器兼容性好的
    # 注意：H.264编码需要openh264库支持
    encoders = [
        ('avc1', '.mp4'),   # H.264 - 浏览器兼容性最好，但需要openh264
        ('mp4v', '.mp4'),   # MPEG-4 - 浏览器兼容性较好
        ('XVID', '.avi'),   # XVID
        ('MJPG', '.avi'),   # Motion JPEG - 最兼容但文件大
    ]
    
    for fourcc_code, ext in encoders:
        try:
            # 生成适合当前编码器的输出路径
            base_path = output_path.rsplit('.', 1)[0]
            encoder_output_path = f"{base_path}{ext}"
            
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            out = cv2.VideoWriter(encoder_output_path, fourcc, fps, (width, height))
            
            if out.isOpened():
                for frame in frames:
                    out.write(frame)
                out.release()
                
                # 检查文件大小
                if os.path.exists(encoder_output_path):
                    file_size = os.path.getsize(encoder_output_path)
                    
                    # 如果文件太大或太小（编码失败），尝试下一个编码器
                    if file_size > 150 * 1024 * 1024 or file_size < 1000:  # 150 MB 或小于1KB
                        os.unlink(encoder_output_path)
                        continue
                    
                    # 返回实际使用的文件路径
                    return encoder_output_path
        except Exception as e:
            # 清理临时文件
            if 'encoder_output_path' in locals() and os.path.exists(encoder_output_path):
                try:
                    os.unlink(encoder_output_path)
                except:
                    pass
            # 继续尝试下一个编码器
            continue
    
    # 如果所有编码器都失败，尝试使用图像序列
    try:
        base_path = output_path.rsplit('.', 1)[0]
        frames_dir = f"{base_path}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.jpg")
            # 降低JPEG质量以减小文件大小
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # 尝试使用ffmpeg创建视频
        import subprocess
        video_path = f"{base_path}.mp4"
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%06d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '28',  # 更高的CRF值 = 更小的文件
            video_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=300)
            # 清理临时帧
            import shutil
            shutil.rmtree(frames_dir)
            return video_path
        except:
            # ffmpeg失败，返回None
            pass
    except:
        pass
    
    return None


def process_video_stream(video_frames, model_path, progress_bar, status_text):
    """
    处理视频流，执行完整的分析流程
    """
    total_frames = len(video_frames)
    
    # 初始化跟踪器
    status_text.text("正在初始化模型...")
    tracker = Tracker(model_path=model_path)
    
    # 获取目标跟踪结果
    status_text.text("正在进行目标检测与跟踪...")
    tracks = tracker.get_object_tracks(video_frames)
    
    # 添加位置信息到跟踪结果
    status_text.text("正在添加位置信息...")
    tracker.add_position_to_tracks(tracks)
    
    # 插值球的位置
    status_text.text("正在插值球的位置...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # 初始化摄像机运动估计器
    status_text.text("正在估计摄像机运动...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)
    
    # 初始化视角变换器
    status_text.text("正在初始化视角变换...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # 初始化速度和距离估计器
    status_text.text("正在计算速度和距离...")
    speed_and_distance_estimator = SpeedAndDistance_Estimate()
    speed_and_distance_estimator.add_seppd_and_distance_to_tracks(tracks)
    
    # 初始化队伍分配器
    status_text.text("正在分配队伍...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    # 初始化球权分配器
    status_text.text("正在分析球权...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks['players']):
        # 获取球的边界框，处理不同数据结构
        ball_bbox = None
        if tracks['ball'][frame_num]:
            ball_track = tracks['ball'][frame_num]
            # 球的数据可能是字典（key为track_id）或直接是包含bbox的字典
            if isinstance(ball_track, dict):
                if 'bbox' in ball_track:
                    ball_bbox = ball_track['bbox']
                elif 1 in ball_track and isinstance(ball_track[1], dict):
                    ball_bbox = ball_track[1].get('bbox')
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    
    team_ball_control = np.array(team_ball_control)
    
    # 绘制标注
    status_text.text("正在生成可视化结果...")
    output_frames = []
    for frame_num in range(total_frames):
        frame = video_frames[frame_num].copy()
        
        # 获取当前帧的跟踪数据
        player_dict = tracks['players'][frame_num]
        referees_dict = tracks['referees'][frame_num]
        ball_dict = tracks['ball'][frame_num]
        
        # 绘制球员
        for track_id, player in player_dict.items():
            color = player.get('team_color', (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player['bbox'], color, track_id, thickness=2)
            if player.get('has_ball', False):
                frame = tracker.draw_traingle(frame, player['bbox'], (0, 0, 255))
        
        # 绘制裁判
        for track_id, referee in referees_dict.items():
            frame = tracker.draw_ellipse(frame, referee['bbox'], (0, 255, 255), track_id, thickness=2)
        
        # 绘制足球
        if isinstance(ball_dict, dict) and 'bbox' in ball_dict and len(ball_dict['bbox']) == 4:
            frame = tracker.draw_traingle(frame, ball_dict['bbox'], (0, 255, 0))
        
        # 绘制摄像机运动
        camera_movement = camera_movement_per_frame[frame_num]
        if isinstance(camera_movement, (list, tuple)) and len(camera_movement) == 2:
            x_movement, y_movement = camera_movement
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=3)
        
        # 绘制速度和距离
        for object_type, object_tracks in tracks.items():
            if object_type == 'ball' or object_type == 'referees':
                continue
            if frame_num < len(object_tracks):
                for track_id, track_info in object_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        distance_covered = track_info.get('distance_covered', None)
                        if speed is None or distance_covered is None:
                            continue
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))
                        cv2.putText(frame, f"Speed:{speed:.1f}km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"Distance:{distance_covered:.1f}m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 绘制球权统计
        frame = tracker.draw_team_ball_control(frame, frame_num, team_ball_control)
        
        output_frames.append(frame)
        
        # 更新进度
        progress = (frame_num + 1) / total_frames
        progress_bar.progress(progress)
    
    status_text.text("分析完成！")
    return output_frames, tracks, team_ball_control


def main():
    """主函数"""
    # 清理之前的临时文件
    cleanup_temp_files()
    
    # 页面标题
    st.title("⚽ 足球比赛智能分析系统")
    st.markdown("基于YOLOv8的目标检测与跟踪系统")
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 配置选项")
    
    # 模型选择
    model_options = {
        "models/soccer_yolov8x2.pt": "YOLOv8x2 (推荐)",
        "models/soccer_yolov11x.pt": "YOLOv11x",
        "models/soccer_yolov12x.pt": "YOLOv12x",
        "models/soccer_yolov8x.pt": "YOLOv8x"
    }
    
    model_path = st.sidebar.selectbox(
        "选择模型",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    # 视频上传
    st.sidebar.subheader("📹 视频上传")
    uploaded_file = st.sidebar.file_uploader(
        "上传足球比赛视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持MP4、AVI、MOV、MKV格式"
    )
    
    # 使用示例视频选项
    use_sample = st.sidebar.checkbox(
        "使用示例视频",
        value=False if uploaded_file else True,
        disabled=uploaded_file is not None
    )
    

    
    # 主页面
    if uploaded_file is not None or use_sample:
        # 获取视频路径
        original_video_path = None  # 用于播放的原始视频路径
        if uploaded_file is not None:
            # 保存上传的文件
            original_video_path = tempfile.mktemp(suffix='.mp4')
            with open(original_video_path, 'wb') as f:
                f.write(uploaded_file.read())
            video_path = original_video_path
        else:
            # 使用示例视频
            video_path = 'input_videos/08fd33_4.mp4'
            original_video_path = video_path
            if not os.path.exists(video_path):
                st.error(f"示例视频不存在: {video_path}")
                st.info("请上传视频文件进行分析")
                return
        
        # 读取视频
        try:
            video_frames = read_video(video_path)
            if not video_frames:
                st.error("无法读取视频文件")
                return
        except Exception as e:
            st.error(f"视频读取失败: {str(e)}")
            return
        
        # 显示视频信息
        st.subheader("📊 视频信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总帧数", len(video_frames))
        with col2:
            st.metric("视频时长", f"{len(video_frames)/24:.1f}秒")
        with col3:
            st.metric("分辨率", f"{video_frames[0].shape[1]}x{video_frames[0].shape[0]}")
        
        # 创建选项卡
        tab_preview, tab_original, tab_result, tab_analysis = st.tabs([
            "📸 预览", "🎬 原视频", "✨ 分析结果", "📈 数据分析"
        ])
        
        with tab_preview:
            # 显示原始视频预览（第一帧）
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始视频预览")
                st.image(video_frames[0], channels="BGR", width="content")
            with col2:
                st.subheader("分析说明")
                st.markdown("""
                点击"开始分析"按钮后，系统将：
                1. 检测球员、裁判和足球
                2. 跟踪目标运动轨迹
                3. 识别两支队伍
                4. 分析球权归属
                5. 计算球员速度和距离
                6. 生成可视化结果
                """)
        
        with tab_original:
            # 显示原视频播放器
            st.subheader("原始视频")
            
            # 直接使用原始视频文件播放
            if original_video_path and os.path.exists(original_video_path):
                # 显示调试信息
                file_size = os.path.getsize(original_video_path)
                st.info(f"视频文件: {original_video_path}")
                st.info(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
                
                # 使用文件路径播放
                try:
                    st.video(original_video_path)
                except Exception as e:
                    st.error(f"视频播放失败: {str(e)}")
                
                # 稍后清理（如果是临时文件）
                if original_video_path != 'input_videos/08fd33_4.mp4':
                    st.session_state['temp_files'] = st.session_state.get('temp_files', []) + [original_video_path]
            else:
                st.error("视频文件不存在")
        
        # 处理按钮
        if st.button("🚀 开始分析", type="primary"):
            # 创建进度显示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 处理视频
                output_frames, tracks, team_control = process_video_stream(
                    video_frames, model_path, progress_bar, status_text
                )
                
                # 切换到结果选项卡并显示
                with tab_result:
                    st.subheader("🎬 分析结果视频")
                    
                    # 显示分析后的视频播放器
                    result_video_path = tempfile.mktemp(suffix='.mp4')
                    actual_result_path = save_video_to_file(output_frames, result_video_path, fps=24)
                    
                    if actual_result_path:
                        # 显示调试信息
                        file_size = os.path.getsize(actual_result_path)
                        st.info(f"分析结果视频文件: {actual_result_path}")
                        st.info(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
                        st.info(f"文件扩展名: {os.path.splitext(actual_result_path)[1]}")
                        
                        # 使用文件路径播放
                        try:
                            st.video(actual_result_path)
                        except Exception as e:
                            st.error(f"视频播放失败: {str(e)}")
                            st.info("尝试使用文件对象播放...")
                            try:
                                with open(actual_result_path, 'rb') as f:
                                    st.video(f)
                            except Exception as e2:
                                st.error(f"文件对象播放也失败: {str(e2)}")
                        
                        # 稍后清理
                        st.session_state['temp_files'] = st.session_state.get('temp_files', []) + [actual_result_path]
                        
                        # 提供下载按钮
                        st.subheader("💾 下载选项")
                        if file_size < 150 * 1024 * 1024:  # 150 MB
                            with open(actual_result_path, 'rb') as f:
                                video_bytes = f.read()
                                st.download_button(
                                    label="📥 下载分析结果视频",
                                    data=video_bytes,
                                    file_name="football_analysis_output.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.warning("视频文件过大，无法提供下载")
                    else:
                        st.error("分析结果视频编码失败")
                    
                    # 显示结果预览图
                    st.subheader("📸 结果预览")
                    col_result1, col_result2 = st.columns(2)
                    with col_result1:
                        st.image(video_frames[0], channels="BGR", caption="原始帧", width="content")
                    with col_result2:
                        st.image(output_frames[0], channels="BGR", caption="分析后帧", width="content")
                
                # 显示统计信息
                st.subheader("📊 分析统计")
                
                # 队伍控球统计
                team1_control = np.sum(team_control == 1)
                team2_control = np.sum(team_control == 2)
                total_control = len(team_control)
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("队伍1控球率", f"{team1_control/total_control*100:.1f}%")
                with col_stat2:
                    st.metric("队伍2控球率", f"{team2_control/total_control*100:.1f}%")
                with col_stat3:
                    st.metric("检测到的球员数", len(set([pid for frame in tracks['players'] for pid in frame.keys()])))
                
                # 数据分析选项卡
                with tab_analysis:
                    st.subheader("📈 数据分析与可视化")
                    
                    # 初始化数据分析器
                    analyzer = FootballDataAnalyzer(tracks, team_control)
                    
                    # 创建子选项卡
                    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                        "控球分析", "球员统计", "热力图", "雷达图", "数据导出"
                    ])
                    
                    with analysis_tab1:
                        st.markdown("### 控球率时间线")
                        possession_chart = analyzer.generate_possession_chart(team_control)
                        if possession_chart:
                            st.plotly_chart(possession_chart, width='stretch')
                        else:
                            st.warning("暂无控球数据")
                        
                        # 队伍统计
                        st.markdown("### 队伍控球统计")
                        team_stats = analyzer.generate_team_statistics(team_control)
                        if team_stats:
                            col_team1, col_team2 = st.columns(2)
                            with col_team1:
                                st.markdown("**队伍1**")
                                st.write(f"控球帧数: {team_stats['team1']['frames']}帧")
                                st.write(f"控球率: {team_stats['team1']['possession_pct']:.1f}%")
                            with col_team2:
                                st.markdown("**队伍2**")
                                st.write(f"控球帧数: {team_stats['team2']['frames']}帧")
                                st.write(f"控球率: {team_stats['team2']['possession_pct']:.1f}%")
                    
                    with analysis_tab2:
                        st.markdown("### 球员数据统计")
                        player_stats = analyzer.calculate_player_statistics(min_appearances=10)
                        
                        if not player_stats.empty:
                            st.info(f"已过滤出现次数少于10帧的球员，共{len(player_stats)}名球员")
                            
                            # 速度对比图表
                            st.markdown("#### 球员速度对比")
                            speed_chart = analyzer.generate_speed_comparison_chart(player_stats)
                            if speed_chart:
                                st.plotly_chart(speed_chart, width='stretch')
                            
                            # 跑动距离图表
                            st.markdown("#### 球员跑动距离")
                            distance_chart = analyzer.generate_distance_comparison(player_stats)
                            if distance_chart:
                                st.plotly_chart(distance_chart, width='stretch')
                            
                            # 球员活动时间线
                            st.markdown("#### 球员活动时间线")
                            timeline_chart = analyzer.generate_player_activity_timeline(player_stats)
                            if timeline_chart:
                                st.plotly_chart(timeline_chart, width='stretch')
                        else:
                            st.warning("暂无球员数据（可能所有球员出现次数都少于10帧）")
                    
                    with analysis_tab3:
                        st.markdown("### 球员活动热力图")
                        
                        # 生成并显示队伍1热力图
                        st.markdown("#### 队伍1热力图")
                        heatmap_team1 = analyzer.generate_heatmap(team_id=1)
                        if heatmap_team1:
                            st.pyplot(heatmap_team1)
                        else:
                            st.warning("队伍1暂无热力图数据")
                        
                        # 生成并显示队伍2热力图
                        st.markdown("#### 队伍2热力图")
                        heatmap_team2 = analyzer.generate_heatmap(team_id=2)
                        if heatmap_team2:
                            st.pyplot(heatmap_team2)
                        else:
                            st.warning("队伍2暂无热力图数据")
                        
                        # 生成并显示所有球员热力图
                        st.markdown("#### 所有球员热力图")
                        heatmap_all = analyzer.generate_heatmap()
                        if heatmap_all:
                            st.pyplot(heatmap_all)
                        else:
                            st.warning("暂无热力图数据")
                    
                    with analysis_tab4:
                        st.markdown("### 队伍雷达图对比")
                        
                        # 队伍对比雷达图
                        st.markdown("#### 队伍综合能力对比")
                        radar_chart = analyzer.generate_team_comparison_radar(player_stats)
                        if radar_chart:
                            st.plotly_chart(radar_chart, width='stretch')
                        else:
                            st.warning("暂无雷达图数据")
                        
                        # 速度分布直方图
                        st.markdown("#### 球员速度分布")
                        speed_dist_chart = analyzer.generate_speed_distribution(player_stats)
                        if speed_dist_chart:
                            st.plotly_chart(speed_dist_chart, width='stretch')
                        else:
                            st.warning("暂无速度分布数据")
                        
                        # 跑动距离饼图
                        st.markdown("#### 队伍跑动距离占比")
                        distance_pie_chart = analyzer.generate_distance_pie_chart(player_stats)
                        if distance_pie_chart:
                            st.plotly_chart(distance_pie_chart, width='stretch')
                        else:
                            st.warning("暂无跑动距离数据")
                    
                    with analysis_tab5:
                        st.markdown("### 数据导出")
                        
                        # 导出数据到本地文件
                        st.info("💾 数据将自动导出到 `output_data` 目录")
                        
                        if st.button("📁 导出分析数据", type="primary"):
                            try:
                                export_paths = analyzer.export_data(team_control)
                                
                                st.success("✅ 数据导出成功！")
                                
                                col_exp1, col_exp2, col_exp3 = st.columns(3)
                                with col_exp1:
                                    st.markdown("**JSON数据**")
                                    st.code(export_paths['json'], language="text")
                                with col_exp2:
                                    st.markdown("**球员统计CSV**")
                                    st.code(export_paths['player_csv'], language="text")
                                with col_exp3:
                                    st.markdown("**队伍统计CSV**")
                                    st.code(export_paths['team_csv'], language="text")
                                
                                # 提供下载按钮
                                st.markdown("#### 下载导出的文件")
                                
                                # 下载JSON
                                with open(export_paths['json'], 'r', encoding='utf-8') as f:
                                    json_data = f.read()
                                st.download_button(
                                    label="📥 下载完整数据 (JSON)",
                                    data=json_data,
                                    file_name=f"match_analysis_{export_paths['json'].split('_')[-1].replace('.json', '')}.json",
                                    mime="application/json"
                                )
                                
                                # 下载球员CSV
                                with open(export_paths['player_csv'], 'rb') as f:
                                    player_csv_data = f.read()
                                st.download_button(
                                    label="📥 下载球员统计 (CSV)",
                                    data=player_csv_data,
                                    file_name=f"player_stats_{export_paths['player_csv'].split('_')[-1].replace('.csv', '')}.csv",
                                    mime="text/csv"
                                )
                                
                                # 下载队伍CSV
                                with open(export_paths['team_csv'], 'rb') as f:
                                    team_csv_data = f.read()
                                st.download_button(
                                    label="📥 下载队伍统计 (CSV)",
                                    data=team_csv_data,
                                    file_name=f"team_stats_{export_paths['team_csv'].split('_')[-1].replace('.csv', '')}.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"导出失败: {str(e)}")
                        
                        # 显示球员统计表格预览
                        st.markdown("#### 球员统计预览")
                        player_stats = analyzer.calculate_player_statistics(min_appearances=10)
                        
                        if not player_stats.empty:
                            display_stats = player_stats[['player_id', 'team', 'appearances', 'total_distance', 'avg_speed', 'max_speed']].copy()
                            display_stats.columns = ['球员ID', '队伍', '出现帧数', '总跑动距离', '平均速度', '最大速度']
                            
                            st.dataframe(
                                display_stats.style.format({
                                    '总跑动距离': '{:.1f}',
                                    '平均速度': '{:.2f}',
                                    '最大速度': '{:.2f}'
                                }),
                                width='stretch'
                            )
                        else:
                            st.warning("暂无球员数据（可能所有球员出现次数都少于10帧）")
                    

                
                st.success("✅ 视频分析完成！")
                
            except Exception as e:
                st.error(f"处理过程中出现错误: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    else:
        # 显示欢迎信息
        st.info("👈 请在侧边栏上传视频文件或选择使用示例视频")
        
        # 显示系统功能介绍
        st.subheader("🎯 系统功能")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **目标检测与跟踪**
            - 球员检测与跟踪
            - 足球检测与跟踪
            - 裁判识别
            - 实时轨迹分析
            """)
        
        with col2:
            st.markdown("""
            **战术分析**
            - 队伍自动识别
            - 球权分析
            - 控球率统计
            - 战术模式识别
            """)
        
        with col3:
            st.markdown("""
            **数据可视化**
            - 球员速度分析
            - 跑动距离统计
            - 活动热力图
            - 控球时间线
            """)
        
        # 显示技术架构
        st.subheader("🔧 技术架构")
        st.markdown("""
        - **目标检测**: YOLOv8 深度学习模型
        - **目标跟踪**: DeepSORT 跟踪算法
        - **队伍识别**: 基于颜色的K-Means聚类
        - **球权分析**: 距离计算与几何分析
        - **速度估计**: 视角变换与运动补偿
        """)


if __name__ == "__main__":
    main()
