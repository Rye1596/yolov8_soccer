#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大模型集成测试脚本
"""
from llm_integration import LLMIntegrator
import pandas as pd
import numpy as np

# 测试大模型集成
def test_llm_integration():
    print("测试大模型集成...")
    
    # 初始化集成器
    integrator = LLMIntegrator()
    
    # 测试获取可用模型
    models = integrator.get_available_models()
    print("可用模型:")
    for model in models:
        print(f"- {model['id']}: {model['name']} ({model['model']})")
    
    # 创建测试数据
    test_player_stats = pd.DataFrame({
        'player_id': [1, 2, 3, 4],
        'team': [1, 1, 2, 2],
        'appearances': [100, 80, 90, 70],
        'total_distance': [1000, 800, 950, 750],
        'avg_speed': [5.2, 4.8, 5.5, 4.5],
        'max_speed': [8.5, 7.8, 9.0, 7.5]
    })
    
    test_tracks = {
        'players': [
            {1: {'team': 1}, 2: {'team': 1}},
            {1: {'team': 1}, 3: {'team': 2}},
            {2: {'team': 1}, 4: {'team': 2}}
        ],
        'ball': [{'bbox': [100, 100, 110, 110]}]
    }
    
    test_team_control = np.array([1, 1, 2, 2, 1])
    
    # 测试分析功能
    print("\n测试分析功能...")
    
    # 构建测试数据
    match_data = {
        "match_info": {
            "视频长度": "100 帧"
        },
        "team_stats": {
            "队伍1": {
                "控球时长": "60 帧",
                "控球率": "60.0%"
            },
            "队伍2": {
                "控球时长": "40 帧",
                "控球率": "40.0%"
            }
        },
        "player_stats": test_player_stats,
        "possession_stats": {
            "队伍1控球率": "60.0%",
            "队伍2控球率": "40.0%"
        }
    }
    
    # 测试格式化功能
    formatted_data = integrator._format_match_data(match_data)
    print("\n格式化数据示例:")
    print(formatted_data[:500] + "...")
    
    print("\n大模型集成测试完成！")

if __name__ == "__main__":
    test_llm_integration()
