#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据分析模块 - 从跟踪结果中提取数据并生成可视化图表
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 设置matplotlib中文字体，支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FootballDataAnalyzer:
    """足球比赛数据分析器"""
    
    def __init__(self, tracks, video_frames):
        """
        初始化分析器
        
        参数:
            tracks: 跟踪结果字典
            video_frames: 视频帧列表
        """
        self.tracks = tracks
        self.video_frames = video_frames
        self.total_frames = len(video_frames)
        
    def extract_player_trajectories(self):
        """
        提取所有球员的轨迹数据
        
        返回:
            dict: 球员ID到轨迹列表的映射
        """
        trajectories = defaultdict(list)
        
        for frame_num, frame_tracks in enumerate(self.tracks['players']):
            for track_id, track_info in frame_tracks.items():
                position = track_info.get('position', (0, 0))
                team = track_info.get('team', 0)
                trajectories[track_id].append({
                    'frame': frame_num,
                    'x': position[0],
                    'y': position[1],
                    'team': team
                })
        
        return dict(trajectories)
    
    def calculate_ball_possession_timeline(self, team_ball_control):
        """
        计算控球率时间线
        
        参数:
            team_ball_control: 控球队伍数组
            
        返回:
            dict: 包含时间线数据和统计信息
        """
        if len(team_ball_control) == 0:
            return None
        
        # 计算每10帧的控球率
        window_size = 10
        possession_timeline = []
        
        for i in range(0, len(team_ball_control), window_size):
            window = team_ball_control[i:i+window_size]
            team1_count = np.sum(window == 1)
            team2_count = np.sum(window == 2)
            total = len(window)
            
            possession_timeline.append({
                'frame': i,
                'team1_possession': team1_count / total * 100 if total > 0 else 0,
                'team2_possession': team2_count / total * 100 if total > 0 else 0
            })
        
        return possession_timeline
    
    def calculate_player_statistics(self, min_appearances=10):
        """
        计算球员统计数据
        
        参数:
            min_appearances: 最小出现次数阈值，低于此值的球员将被过滤
            
        返回:
            DataFrame: 球员统计数据
        """
        stats = []
        
        for track_id in set([
            tid for frame in self.tracks['players'] 
            for tid in frame.keys()
        ]):
            # 统计该球员的出现次数
            appearances = 0
            total_distance = 0
            speeds = []
            team = None
            positions = []
            
            prev_position = None
            
            for frame_num, frame_tracks in enumerate(self.tracks['players']):
                if track_id in frame_tracks:
                    track_info = frame_tracks[track_id]
                    appearances += 1
                    
                    if team is None:
                        team = track_info.get('team', 0)
                    
                    position = track_info.get('position', None)
                    if position:
                        positions.append(position)
                        
                        if prev_position:
                            distance = np.sqrt(
                                (position[0] - prev_position[0])**2 + 
                                (position[1] - prev_position[1])**2
                            )
                            total_distance += distance
                            speed = distance * 24  # 假设24fps
                            speeds.append(speed)
                        
                        prev_position = position
            
            # 过滤出现次数过少的球员
            if appearances < min_appearances:
                continue
            
            avg_speed = np.mean(speeds) if speeds else 0
            max_speed = np.max(speeds) if speeds else 0
            
            stats.append({
                'player_id': track_id,
                'team': team,
                'appearances': appearances,
                'total_distance': total_distance,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'positions': positions
            })
        
        return pd.DataFrame(stats)
    
    def generate_possession_chart(self, team_ball_control):
        """
        生成控球率图表
        
        参数:
            team_ball_control: 控球队伍数组
            
        返回:
            plotly Figure对象
        """
        if len(team_ball_control) == 0:
            return None
        
        timeline = self.calculate_ball_possession_timeline(team_ball_control)
        if not timeline:
            return None
        
        df = pd.DataFrame(timeline)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df['team1_possession'],
            name='队伍1',
            mode='lines',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            x=df['frame'],
            y=df['team2_possession'],
            name='队伍2',
            mode='lines',
            line=dict(color='#4ECDC4', width=2),
            fill='tozeroy',
            opacity=0.6
        ))
        
        fig.update_layout(
            title='控球率时间线',
            xaxis_title='帧数',
            yaxis_title='控球率 (%)',
            hovermode='x unified',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')  # 添加中文字体支持
        )
        
        return fig
    
    def generate_speed_comparison_chart(self, player_stats):
        """
        生成球员速度对比图表
        
        参数:
            player_stats: 球员统计数据DataFrame
            
        返回:
            plotly Figure对象
        """
        if player_stats.empty:
            return None
        
        # 按队伍分组
        team1 = player_stats[player_stats['team'] == 1]
        team2 = player_stats[player_stats['team'] == 2]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('队伍1球员速度', '队伍2球员速度')
        )
        
        if not team1.empty:
            fig.add_trace(
                go.Bar(
                    x=team1['player_id'],
                    y=team1['avg_speed'],
                    name='队伍1',
                    marker_color='#FF6B6B'
                ),
                row=1, col=1
            )
        
        if not team2.empty:
            fig.add_trace(
                go.Bar(
                    x=team2['player_id'],
                    y=team2['avg_speed'],
                    name='队伍2',
                    marker_color='#4ECDC4'
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="球员ID", row=1, col=1)
        fig.update_xaxes(title_text="球员ID", row=1, col=2)
        fig.update_yaxes(title_text="平均速度 (像素/秒)", row=1, col=1)
        fig.update_yaxes(title_text="平均速度 (像素/秒)", row=1, col=2)
        
        fig.update_layout(
            title='球员速度对比',
            showlegend=False,
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')  # 添加中文字体支持
        )
        
        return fig
    
    def generate_heatmap(self, team_id=None):
        """
        生成球员活动热力图
        
        参数:
            team_id: 队伍ID（可选，None表示所有球员）
            
        返回:
            matplotlib Figure对象
        """
        # 收集所有位置数据
        positions = []
        
        for frame_tracks in self.tracks['players']:
            for track_id, track_info in frame_tracks.items():
                if team_id is None or track_info.get('team') == team_id:
                    position = track_info.get('position', None)
                    if position:
                        positions.append(position)
        
        if not positions:
            return None
        
        positions = np.array(positions)
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 使用kdeplot绘制热力图
        try:
            sns.kdeplot(
                x=positions[:, 0],
                y=positions[:, 1],
                cmap='YlOrRd',
                fill=True,
                thresh=0.05,
                levels=20,
                ax=ax
            )
        except:
            # 如果kdeplot失败，使用hexbin
            ax.hexbin(positions[:, 0], positions[:, 1], gridsize=30, cmap='YlOrRd')
        
        title_text = f'球员活动热力图 - {"队伍" + str(team_id) if team_id else "所有球员"}'
        ax.set_title(title_text)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        return fig
    
    def generate_distance_comparison(self, player_stats):
        """
        生成球员跑动距离对比图表
        
        参数:
            player_stats: 球员统计数据DataFrame
            
        返回:
            plotly Figure对象
        """
        if player_stats.empty:
            return None
        
        fig = px.bar(
            player_stats.sort_values('total_distance', ascending=True),
            x='player_id',
            y='total_distance',
            color='team',
            color_discrete_map={1: '#FF6B6B', 2: '#4ECDC4'},
            title='球员跑动距离对比',
            labels={
                'player_id': '球员ID',
                'total_distance': '总跑动距离 (像素)',
                'team': '队伍'
            }
        )
        
        fig.update_layout(
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')  # 添加中文字体支持
        )
        
        return fig
    
    def generate_team_statistics(self, team_ball_control):
        """
        生成队伍统计信息
        
        参数:
            team_ball_control: 控球队伍数组
            
        返回:
            dict: 队伍统计数据
        """
        if len(team_ball_control) == 0:
            return None
        
        team1_frames = np.sum(team_ball_control == 1)
        team2_frames = np.sum(team_ball_control == 2)
        total_frames = len(team_ball_control)
        
        return {
            'team1': {
                'possession_pct': team1_frames / total_frames * 100,
                'frames': team1_frames
            },
            'team2': {
                'possession_pct': team2_frames / total_frames * 100,
                'frames': team2_frames
            },
            'total_frames': total_frames
        }
    
    def generate_all_analytics(self, team_ball_control):
        """
        生成所有分析图表和数据
        
        参数:
            team_ball_control: 控球队伍数组
            
        返回:
            dict: 包含所有分析结果的字典
        """
        # 提取数据
        trajectories = self.extract_player_trajectories()
        player_stats = self.calculate_player_statistics()
        team_stats = self.generate_team_statistics(team_ball_control)
        
        # 生成图表
        charts = {
            'possession_chart': self.generate_possession_chart(team_ball_control),
            'speed_chart': self.generate_speed_comparison_chart(player_stats),
            'distance_chart': self.generate_distance_comparison(player_stats),
            'heatmap_all': self.generate_heatmap(),
            'heatmap_team1': self.generate_heatmap(team_id=1),
            'heatmap_team2': self.generate_heatmap(team_id=2)
        }
        
        return {
            'trajectories': trajectories,
            'player_stats': player_stats,
            'team_stats': team_stats,
            'charts': charts
        }
    
    def export_data(self, team_ball_control, output_dir='output_data'):
        """
        导出分析数据到本地文件
        
        参数:
            team_ball_control: 控球队伍数组
            output_dir: 输出目录
            
        返回:
            dict: 导出的文件路径
        """
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取数据
        trajectories = self.extract_player_trajectories()
        player_stats = self.calculate_player_statistics()
        team_stats = self.generate_team_statistics(team_ball_control)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 导出为JSON
        json_filename = f"match_analysis_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        
        export_data = {
            'match_info': {
                'total_frames': self.total_frames,
                'video_duration': f"{self.total_frames / 24:.1f}秒",
                'analysis_time': datetime.now().isoformat()
            },
            'team_stats': team_stats,
            'player_stats': player_stats.to_dict('records') if not player_stats.empty else [],
            'trajectories': trajectories
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        # 导出球员统计为CSV
        csv_filename = f"player_stats_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        if not player_stats.empty:
            display_stats = player_stats[['player_id', 'team', 'appearances', 'total_distance', 'avg_speed', 'max_speed']].copy()
            display_stats.columns = ['球员ID', '队伍', '出现帧数', '总跑动距离(像素)', '平均速度(像素/秒)', '最大速度(像素/秒)']
            display_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 导出队伍统计为CSV
        team_csv_filename = f"team_stats_{timestamp}.csv"
        team_csv_path = os.path.join(output_dir, team_csv_filename)
        
        if team_stats:
            team_data = []
            for team_id in [1, 2]:
                team_name = f"队伍{team_id}"
                team_key = f'team{team_id}'
                if team_key in team_stats:
                    team_data.append({
                        '队伍': team_name,
                        '控球帧数': team_stats[team_key]['frames'],
                        '控球率(%)': team_stats[team_key]['possession_pct']
                    })
            
            if team_data:
                team_df = pd.DataFrame(team_data)
                team_df.to_csv(team_csv_path, index=False, encoding='utf-8-sig')
        
        return {
            'json': json_path,
            'player_csv': csv_path,
            'team_csv': team_csv_path
        }
    
    def generate_team_comparison_radar(self, player_stats):
        """
        生成队伍对比雷达图
        
        参数:
            player_stats: 球员统计数据DataFrame
            
        返回:
            plotly Figure对象
        """
        if player_stats.empty:
            return None
        
        # 按队伍分组计算平均指标
        team1 = player_stats[player_stats['team'] == 1]
        team2 = player_stats[player_stats['team'] == 2]
        
        if team1.empty or team2.empty:
            return None
        
        # 计算各项指标
        metrics = {
            '平均速度': [team1['avg_speed'].mean(), team2['avg_speed'].mean()],
            '最大速度': [team1['max_speed'].mean(), team2['max_speed'].mean()],
            '平均跑动距离': [team1['total_distance'].mean(), team2['total_distance'].mean()],
            '球员平均出现次数': [team1['appearances'].mean(), team2['appearances'].mean()]
        }
        
        # 归一化数据
        normalized_metrics = {}
        for metric, values in metrics.items():
            max_val = max(values)
            if max_val > 0:
                normalized_metrics[metric] = [v / max_val for v in values]
            else:
                normalized_metrics[metric] = values
        
        fig = go.Figure()
        
        categories = list(normalized_metrics.keys())
        fig.add_trace(go.Scatterpolar(
            r=normalized_metrics['平均速度'],
            theta=categories,
            fill='toself',
            name='队伍1',
            line_color='#FF6B6B'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_metrics['最大速度'],
            theta=categories,
            fill='toself',
            name='队伍2',
            line_color='#4ECDC4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='队伍综合能力对比',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')
        )
        
        return fig
    
    def generate_player_activity_timeline(self, player_stats):
        """
        生成球员活动时间线图
        
        参数:
            player_stats: 球员统计数据DataFrame
            
        返回:
            plotly Figure对象
        """
        if player_stats.empty:
            return None
        
        # 按队伍和球员ID排序
        player_stats = player_stats.sort_values(['team', 'player_id'])
        
        fig = go.Figure()
        
        for team_id in [1, 2]:
            team_data = player_stats[player_stats['team'] == team_id]
            if not team_data.empty:
                fig.add_trace(go.Bar(
                    x=team_data['player_id'].astype(str),
                    y=team_data['appearances'],
                    name=f'队伍{team_id}',
                    marker_color='#FF6B6B' if team_id == 1 else '#4ECDC4'
                ))
        
        fig.update_layout(
            title='球员活动时间线（出现帧数）',
            xaxis_title='球员ID',
            yaxis_title='出现帧数',
            barmode='group',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')
        )
        
        return fig
    
    def generate_speed_distribution(self, player_stats):
        """
        生成速度分布直方图
        
        参数:
            player_stats: 球员统计数据DataFrame
            
        返回:
            plotly Figure对象
        """
        if player_stats.empty:
            return None
        
        fig = go.Figure()
        
        for team_id in [1, 2]:
            team_data = player_stats[player_stats['team'] == team_id]
            if not team_data.empty:
                fig.add_trace(go.Histogram(
                    x=team_data['avg_speed'],
                    name=f'队伍{team_id}',
                    opacity=0.7,
                    marker_color='#FF6B6B' if team_id == 1 else '#4ECDC4',
                    nbinsx=20
                ))
        
        fig.update_layout(
            title='球员平均速度分布',
            xaxis_title='平均速度 (像素/秒)',
            yaxis_title='球员数量',
            barmode='overlay',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')
        )
        
        return fig
    
    def generate_distance_pie_chart(self, player_stats):
        """
        生成跑动距离饼图
        
        参数:
            player_stats: 球员统计数据DataFrame
            
        返回:
            plotly Figure对象
        """
        if player_stats.empty:
            return None
        
        # 按队伍分组计算总跑动距离
        team1_distance = player_stats[player_stats['team'] == 1]['total_distance'].sum()
        team2_distance = player_stats[player_stats['team'] == 2]['total_distance'].sum()
        
        if team1_distance == 0 and team2_distance == 0:
            return None
        
        fig = go.Figure(data=[go.Pie(
            labels=['队伍1', '队伍2'],
            values=[team1_distance, team2_distance],
            marker=dict(colors=['#FF6B6B', '#4ECDC4']),
            hole=0.3
        )])
        
        fig.update_layout(
            title='队伍跑动距离占比',
            template='plotly_white',
            font=dict(family='Microsoft YaHei, SimHei, Arial')
        )
        
        return fig
