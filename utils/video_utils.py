import cv2

def read_video(video_path):
    """
    读取视频文件并返回所有帧的列表
    :param video_path: 视频文件路径
    :return: 包含所有帧的列表
    """
    # 打开视频文件
    clip = cv2.VideoCapture(video_path)
    frames = []
    # 逐帧读取视频
    while True:
        ret, frame = clip.read()
        if not ret:
            # 读取失败时退出循环
            break
        frames.append(frame)
    # 返回所有帧
    return frames


def write_video(output_video_frames, output_video_path):
    """
    将帧列表写入视频文件
    :param output_video_frames: 要写入的帧列表
    :param output_video_path: 输出视频文件路径
    """
    # 定义视频编码格式为XVID
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 创建VideoWriter对象，帧率为24，尺寸与第一帧相同
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    # 逐帧写入视频
    for frame in output_video_frames:
        out.write(frame)
    # 释放VideoWriter资源
    out.release()