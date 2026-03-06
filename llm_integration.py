#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型集成模块 - 支持阿里通义千问进行足球比赛智能分析
"""
import os
import json
import time
import requests
from typing import Dict, Optional, List
import pandas as pd


class LLMIntegrator:
    """
    大模型集成器，支持阿里通义千问API
    """
    
    def __init__(self):
        """
        初始化大模型集成器
        """
        self.models = {
            "qwen": {
                "name": "阿里通义千问",
                "api_url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "model": "qwen-turbo"
            }
        }
    
    def get_available_models(self) -> List[Dict]:
        """
        获取可用的大模型列表
        
        返回:
            List[Dict]: 模型信息列表
        """
        return [
            {
                "id": model_id,
                "name": model_info["name"],
                "model": model_info["model"]
            }
            for model_id, model_info in self.models.items()
        ]
    
    def generate_analysis(self, 
                         model_id: str, 
                         api_key: str, 
                         match_data: Dict, 
                         max_tokens: int = 1500,
                         temperature: float = 0.7) -> Optional[str]:
        """
        使用通义千问生成比赛分析
        
        参数:
            model_id: 模型ID
            api_key: API密钥
            match_data: 比赛数据
            max_tokens: 最大生成token数
            temperature: 生成温度
            
        返回:
            Optional[str]: 分析结果
        """
        if model_id not in self.models:
            return "错误: 不支持的模型"
        
        model_info = self.models[model_id]
        
        try:
            return self._call_qwen(
                api_key=api_key,
                model=model_info["model"],
                match_data=match_data,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            return f"API调用失败: {str(e)}"
    
    def _call_qwen(self, api_key: str, model: str, match_data: Dict, max_tokens: int, temperature: float) -> str:
        """
        调用阿里通义千问API
        
        文档: https://help.aliyun.com/zh/dashscope/developer-reference/api-details
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的足球比赛分析师，擅长分析比赛数据并提供专业的战术分析。请基于提供的数据，生成简短、精炼的比赛分析报告，重点突出关键信息和核心结论，使用中文回答。"
            },
            {
                "role": "user",
                "content": self.format_match_data(match_data)
            }
        ]
        
        payload = {
            "model": model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "result_format": "message"
            }
        }
        
        response = requests.post(
            self.models["qwen"]["api_url"],
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        data = response.json()
        
        # 解析通义千问的响应
        if "output" in data and "choices" in data["output"] and len(data["output"]["choices"]) > 0:
            return data["output"]["choices"][0]["message"]["content"]
        else:
            return f"API响应格式错误: {json.dumps(data, ensure_ascii=False)}"
    
    def format_match_data(self, match_data: Dict) -> str:
        """
        格式化比赛数据为自然语言
        
        参数:
            match_data: 比赛数据
            
        返回:
            str: 格式化后的文本
        """
        parts = []
        
        # 基本信息
        if "match_info" in match_data:
            parts.append("## 比赛基本信息")
            for key, value in match_data["match_info"].items():
                parts.append(f"- {key}: {value}")
        
        # 队伍统计
        if "team_stats" in match_data:
            parts.append("\n## 队伍统计")
            for team, stats in match_data["team_stats"].items():
                parts.append(f"### {team}")
                for key, value in stats.items():
                    parts.append(f"- {key}: {value}")
        
        # 球员统计
        if "player_stats" in match_data:
            parts.append("\n## 球员统计")
            player_stats = match_data["player_stats"]
            if isinstance(player_stats, pd.DataFrame):
                # 转换为文本
                parts.append("### 球员数据")
                for _, row in player_stats.iterrows():
                    player_info = []
                    for col in player_stats.columns:
                        if col != 'positions':
                            player_info.append(f"{col}: {row[col]}")
                    parts.append(f"- 球员 {row.get('player_id', 'N/A')}: {', '.join(player_info)}")
            else:
                parts.append(str(player_stats))
        
        # 控球率
        if "possession_stats" in match_data:
            parts.append("\n## 控球率统计")
            parts.append(str(match_data["possession_stats"]))
        
        # 分析要求
        parts.append("\n## 分析要求")
        parts.append("1. 基于以上数据，分析比赛的整体态势")
        parts.append("2. 分析两支队伍的战术特点和表现")
        parts.append("3. 分析关键球员的表现和作用")
        parts.append("4. 提供具体的战术建议")
        parts.append("5. 生成简短、精炼的分析报告，重点突出关键信息，使用中文")
        
        return "\n".join(parts)
    
    def analyze_match(self, 
                     model_id: str, 
                     api_key: str, 
                     tracks: Dict, 
                     team_control: List, 
                     player_stats: pd.DataFrame,
                     video_info: Optional[Dict] = None) -> Optional[str]:
        """
        分析比赛数据并生成智能分析
        
        参数:
            model_id: 模型ID
            api_key: API密钥
            tracks: 跟踪结果
            team_control: 控球队伍数组
            player_stats: 球员统计数据
            video_info: 视频信息
            
        返回:
            Optional[str]: 分析结果
        """
        # 构建比赛数据
        match_data = {
            "match_info": {
                "视频长度": f"{len(tracks['players'])} 帧"
            }
        }
        
        # 计算队伍统计
        if len(team_control) > 0:
            team1_frames = sum(1 for t in team_control if t == 1)
            team2_frames = sum(1 for t in team_control if t == 2)
            total_frames = len(team_control)
            
            match_data["team_stats"] = {
                "队伍1": {
                    "控球时长": f"{team1_frames} 帧",
                    "控球率": f"{(team1_frames / total_frames * 100):.1f}%"
                },
                "队伍2": {
                    "控球时长": f"{team2_frames} 帧",
                    "控球率": f"{(team2_frames / total_frames * 100):.1f}%"
                }
            }
            
            match_data["possession_stats"] = {
                "队伍1控球率": f"{(team1_frames / total_frames * 100):.1f}%",
                "队伍2控球率": f"{(team2_frames / total_frames * 100):.1f}%"
            }
        
        # 添加球员统计
        if not player_stats.empty:
            match_data["player_stats"] = player_stats
        
        # 调用大模型
        return self.generate_analysis(
            model_id=model_id,
            api_key=api_key,
            match_data=match_data
        )
