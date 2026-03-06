from utils import read_video, write_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camear_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimate import SpeedAndDistance_Estimate

def main():
    # 读取输入视频
    video_path = "input_videos/08fd33_4.mp4"
    video_frames = read_video(video_path)
    # 提取视频文件名（不含扩展名）
    video_name = video_path.split("/")[-1].split(".")[0]

    # 初始化目标检测与跟踪器
    model_path = 'models/soccer_yolov8x2.pt'
    tracker = Tracker(model_path=model_path)
    model_name = model_path.split("/")[-1].split(".")[0]

    # 获取目标跟踪结果，优先读取本地缓存
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=f'stubs/track_stubs_{video_name}_{model_name}.pkl'
    )

    # 插值球的位置，必须在计算球的位置之前完成，否则位置为空会导致距离计算报错
    tracks["ball"] = tracker.interplate_ball_positions(tracks["ball"])

    # 为每个目标添加位置信息（bbox下边界中点），用于后续速度计算
    tracker.add_position_to_tracks(tracks)

    # 摄像机运动估计：计算每帧的摄像机位移
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=f'stubs/camera_movement_{video_name}_{model_name}.pkl'
    )
    # 根据摄像机位移修正目标位置
    camera_movement_estimator.add_adjust_position_to_tracks(tracks, camera_movement_per_frame)

    # 视角变换：将目标位置映射到俯视图
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 速度与距离估算
    speed_and_distance_estimator = SpeedAndDistance_Estimate()
    speed_and_distance_estimator.add_seppd_and_distance_to_tracks(tracks)

    # 球员分队：根据颜色为球员分配队伍
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # 为每一帧的每个球员分配队伍与对应颜色
    for frame_num, player_tracks in enumerate(tracks['players']):
        for track_id, track in player_tracks.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                track_id
            )
            tracks['players'][frame_num][track_id]['team'] = team
            tracks['players'][frame_num][track_id]['color'] = team_assigner.team_colors[team]

    # 球权分配：判断当前帧哪个球员控球
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks["ball"][frame_num]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # 若无人控球，则沿用上一帧的控球队伍
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # 绘制球员、球及各类信息到输出帧
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # 在帧上绘制摄像机运动箭头
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    # 在帧上绘制球员速度与距离
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames,
        tracks
    )

    # 保存最终输出视频
    output_video_path = f"output_videos/{video_name}_{model_name}_output.avi"
    write_video(output_video_frames, output_video_path)


if __name__ == "__main__":
    main()
    print("done")
