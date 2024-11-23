import cv2
import mediapipe as mp
import os
import json
import numpy as np

# 初始化 MediaPipe 和 OpenCV GPU 模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
print(mp_pose.POSE_CONNECTIONS)
def process_video(video_path, output_dir):
    """
    处理单个视频文件并保存骨架的2D坐标。
    :param video_path: 输入视频文件的路径
    :param output_dir: 输出关节坐标保存文件的目录
    """
    # 使用 OpenCV 读取视频（CUDA 加速）
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    # 验证是否成功打开视频
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    frame_data = {}
    frame_count = 0
    while True:
        # 使用 CUDA 加速读取帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理每帧并识别骨架
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # 提取2D坐标
            landmarks = results.pose_landmarks.landmark
            pose_2d_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in landmarks]

            # 记录关节坐标
            frame_data[frame_count] = pose_2d_coords

        frame_count += 1

    cap.release()

    # 保存关节坐标到 JSON 文件
    if frame_data:
        save_path = os.path.join(output_dir, 'pose_data.json')
        with open(save_path, 'w') as f:
            json.dump(frame_data, f, indent=4)

def process_all_videos_in_folder(input_folder, output_folder):
    """
    处理文件夹中的所有视频文件并保存其骨架的2D坐标。
    :param input_folder: 输入文件夹路径，包含待处理的视频文件
    :param output_folder: 输出文件夹路径，用于保存处理后的数据
    """
    # 遍历文件夹中的所有子文件夹和视频文件
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                video_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path, os.path.splitext(filename)[0])

                # 创建输出文件夹
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 处理视频
                process_video(video_path, output_dir)

# 设置输入和输出文件夹路径
input_folder = "/media/szm/Lenovo PS10/软件工程/AI技术赛道（更新）/训练数据集"
output_folder = "/media/szm/Lenovo PS10/ruanjian"

# 处理文件夹中的所有视频
process_all_videos_in_folder(input_folder, output_folder)
