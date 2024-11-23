import os
import json
import math
import numpy as np
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# 根据时间戳转换为对应的帧索引

def parse_timecode_to_frame(timecode_str, fps):
    """
    将时间码（时:分:秒 帧）转换为对应的帧索引。
    参数:
        timecode_str (str): 时间码字符串，格式为 "hh:mm:ss frame"
        fps (int): 每秒帧数，用于计算帧索引
    返回:
        frame_index (int): 对应的帧索引
    """
    time_part, frame_within_second_str = timecode_str.strip().split(' ')
    hours_str, minutes_str, seconds_str = time_part.strip().split(':')
    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    frame_within_second = int(frame_within_second_str)

    # 计算总秒数
    total_seconds = hours * 3600 + minutes * 60 + seconds

    # 计算帧索引：总秒数 * fps + 当前帧
    frame_index = total_seconds * fps + frame_within_second
    return frame_index
# 动态计算每个关键帧的帧数分配
def calculate_frame_sampling_for_video(video_data, max_time_steps):
    """
    计算每个关键帧的帧数分配，确保总帧数不超过 max_time_steps。
    参数:
        video_data (list): 视频帧数据
        max_time_steps (int): 最大采样帧数
    返回:
        frame_counts (list): 每个关键帧对应的帧数
    """
    num_keyframes = len(video_data)
    avg_frames_per_keyframe = max_time_steps // num_keyframes
    remaining_frames = max_time_steps % num_keyframes
    frame_counts = [avg_frames_per_keyframe] * num_keyframes
    # 为前几帧分配额外的帧
    for i in range(remaining_frames):
        frame_counts[i] += 1
    return frame_counts



def sample_frames_for_keyframes(video_data, first_keyframe, reference_frames, fps, max_time_steps, frame_counts):
    """
    根据关键帧选择前后帧，确保每个关键帧前后的帧数和视频总帧数分配一致。

    参数：
        video_data (list): 视频数据，按帧存储。
        first_keyframe (str): 第一个关键帧的时间戳。
        reference_frames (list): 参考帧时间戳列表。
        fps (int): 帧率。
        max_time_steps (int): 最大时间步数（最大帧数）。
        frame_counts (list): 每个关键帧采样的帧数。

    返回：
        list: 采样的帧索引列表。
    """
    first_keyframe_index = parse_timecode_to_frame(first_keyframe, fps)
    reference_frame_indices = [parse_timecode_to_frame(ref, fps) for ref in reference_frames]

    # 计算采样的时间窗口
    sample_frames = []
    frame_idx = 0  # 当前帧索引

    for idx, ref_frame_index in enumerate(reference_frame_indices):
        # 获取当前参考帧对应的采样数量
        sample_count = frame_counts[idx]

        # 计算前1000ms和后300ms的采样窗口
        start_index = max(0, ref_frame_index - int(1000 * fps / 1000))  # 前1000ms的帧索引
        end_index = min(len(video_data), ref_frame_index + int(300 * fps / 1000))  # 后300ms的帧索引

        # 计算在该时间窗口内均匀采样的帧数
        total_frames_in_window = end_index - start_index
        if total_frames_in_window > 0:
            # 每个时间窗口中均匀采样 sample_count 帧
            sampled_indices_in_window = np.linspace(start_index, end_index - 1, sample_count, dtype=int)
            sample_frames.extend(sampled_indices_in_window)

    # 确保总的采样帧数不超过 max_time_steps
    if len(sample_frames) > max_time_steps:
        sample_frames = random.sample(sample_frames, max_time_steps)

    return sorted(sample_frames)


# 数据增强函数
def augment_data(video_data):
    print("进行数据增强")
    angle = random.uniform(-5, 5)
    rotation_matrix = np.array([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle))],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle))]
    ])
    video_data = np.dot(video_data, rotation_matrix.T)
    noise = np.random.normal(0, 0.01, video_data.shape)
    video_data += noise
    return video_data

def augment_time(video_data, max_time_steps):
    num_time_steps = video_data.shape[0]
    if num_time_steps < max_time_steps:
        pad_width = ((0, max_time_steps - num_time_steps), (0, 0), (0, 0))
        video_data = np.pad(video_data, pad_width, mode='constant')
    elif num_time_steps > max_time_steps:
        start = random.randint(0, num_time_steps - max_time_steps)
        video_data = video_data[start:start + max_time_steps]
    return video_data


def align_and_normalize_time(video_data, max_time_steps):
    num_time_steps = video_data.shape[0]
    if num_time_steps > max_time_steps:
        video_data = video_data[:max_time_steps, :, :]
    elif num_time_steps < max_time_steps:
        pad_width = ((0, max_time_steps - num_time_steps), (0, 0), (0, 0))
        video_data = np.pad(video_data, pad_width, mode='constant')
    max_value = np.max(video_data)
    if max_value > 0:
        video_data = video_data / max_value
    return video_data

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# 计算关键关节角度
def calculate_angles(video_data):
    """
    关节点编号：
    11 - 左肩 (left shoulder)
    13 - 左肘 (left elbow)
    15 - 左腕 (left wrist)
    12 - 右肩 (right shoulder)
    14 - 右肘 (right elbow)
    16 - 右腕 (right wrist)
    23 - 左髋 (left hip)
    25 - 左膝 (left knee)
    27 - 左踝 (left ankle)
    24 - 右髋 (right hip)
    26 - 右膝 (right knee)
    28 - 右踝 (right ankle)
    """
    angles = []
    for frame in video_data:
        # 计算肩部、肘部、胯部和膝盖的角度
        left_shoulder = calculate_angle(frame[13], frame[11], frame[23])  # 左肩
        right_shoulder = calculate_angle(frame[14], frame[12], frame[24])  # 右肩
        left_hip = calculate_angle(frame[11], frame[23], frame[25])        # 左髋
        right_hip = calculate_angle(frame[12], frame[24], frame[26])       # 右髋
        left_elbow = calculate_angle(frame[11], frame[13], frame[15])      # 左肘
        right_elbow = calculate_angle(frame[12], frame[14], frame[16])     # 右肘
        left_knee = calculate_angle(frame[23], frame[25], frame[27])       # 左膝
        right_knee = calculate_angle(frame[24], frame[26], frame[28])      # 右膝

        # 将角度添加到数组中
        angles.append([left_shoulder, right_shoulder, left_elbow, right_elbow, left_hip, right_hip, left_knee, right_knee])

    # 打印角度数组的形状
    print("计算完成 - 角度数组的形状:", np.array(angles).shape)
    return np.array(angles)  # (num_time_steps, num_angles)

def load_standard_info(mark_dir):
    standard_info_file = os.path.join(mark_dir, 'standard_info.json')
    if not os.path.exists(standard_info_file):
        print(f"未找到 standard_info.json 文件: {standard_info_file}")
        return {}
    with open(standard_info_file, 'r', encoding='utf-8') as f:
        standard_info_list = json.load(f)
    standard_info_dict = {}
    for info in standard_info_list:
        label = int(info['label'])
        standard_info_dict[label] = info
    return standard_info_dict


def store_standard_angles(standard_info_dict , video_data, max_time_steps=100):
    standard_angles = {}

    for action_label, standard_info in standard_info_dict.items():
        video_info = standard_info.get('videoInfo', {})
        fps = int(video_info.get('fps', 30))
        reference_frames = standard_info.get('standardSocreInfo', {}).get('referenceFrames', [])
        first_keyframe = standard_info.get('standardSocreInfo', {}).get('firstKeyFrame', '')



        if not video_data:
            continue

        # Calculate frame counts for sampling
        frame_counts = calculate_frame_sampling_for_video(video_data, max_time_steps)

        # Sample frames based on the keyframe information
        sampled_frames = sample_frames_for_keyframes(video_data, first_keyframe, reference_frames, fps, max_time_steps,
                                                     frame_counts)

        # Extract angles for the sampled frames
        angles = calculate_angles(video_data[sampled_frames])

        # Store angles for each video in the standard_angles dictionary
        standard_angles[action_label] = {
            'keyframe_angles': angles.tolist(),  # Storing as list for easier use
            'reference_frames': reference_frames,
            'first_keyframe': first_keyframe
        }

    return standard_angles


# 数据加载与预处理函数
def load_skeleton_data_with_angles(data_dir, mark_dir, augment=False, max_time_steps=100):
    # 加载标准信息
    standard_info_dict = load_standard_info(mark_dir)
    all_data = []
    all_angles = []
    class_labels = []
    score_labels = []
    key_labels = []
    video_ids = []

    print("开始加载数据...")
    for action_label in os.listdir(data_dir):
        action_path = os.path.join(data_dir, action_label)
        if not os.path.isdir(action_path):
            continue

        # 提取动作编号部分
        #action_number = action_label.split('-')[0]  # 获取 `-` 前面的编号部分
        action_number = re.match(r'^\d+', action_label)
        action_number = action_number.group(0) if action_number else action_label  # 获取匹配到的数字部分
        try:
            action_number = int(action_number)
        except ValueError:
            print(f"跳过无效的动作标签: {action_label}")
            continue

        print(f"处理动作类别: {action_label} (编号: {action_number})")
        # 获取该动作的标准信息
        standard_info = standard_info_dict.get(action_number, None)
        if standard_info is None:
            print(f"未找到动作编号 {action_number} 的标准信息")

            #continue

        # 获取关键帧和fps
        if standard_info:
            reference_frames = standard_info.get('standardSocreInfo', {}).get('referenceFrames', [])
            video_info = standard_info.get('videoInfo', {})
            fps = int(video_info.get('fps', 30))
        """
        # 解析关键帧为帧索引
        reference_frame_indices = []
        for ref_frame in reference_frames:
            frame_index = parse_timecode_to_frame(ref_frame, fps)
            reference_frame_indices.append(frame_index)
        """
        for video_folder in os.listdir(action_path):
            video_path = os.path.join(action_path, video_folder)
            pose_file = os.path.join(video_path, 'pose_data.json')
            if not os.path.exists(pose_file):
                print(f"未找到 pose_data.json 文件: {pose_file}")
                continue

            with open(pose_file, 'r') as f:
                pose_data = json.load(f)

            frame_indices = sorted(pose_data.keys(), key=lambda x: int(x))
            video_data = []
            frame_idx_to_data = {}  # 建立帧索引到数据的映射
            for frame_idx in frame_indices:
                frame_data = pose_data[frame_idx]
                if isinstance(frame_data, dict) and 'joints' in frame_data:
                    frame_data = frame_data['joints']
                elif isinstance(frame_data, list):
                    pass  # 直接使用 frame_data 作为关键点信息
                else:
                    print(f"跳过无效的 frame_data 格式: {frame_data}")
                    continue
                frame_array = np.array([joint[:2] for joint in frame_data if isinstance(joint, list) and len(joint) >= 2])
                if frame_array.shape[0] != 33:
                    print(f"跳过无效的帧，frame_array 的形状为 {frame_array.shape}")
                    continue
                video_data.append(frame_array)
                frame_idx_to_data[frame_idx] = frame_array
            """ 
            # 只提取关键帧的数据
            key_frame_data = []
            for ref_frame_idx in reference_frame_indices:
                # 找到与关键帧索引最接近的实际帧索引
                closest_frame_idx = min(frame_indices, key=lambda x: abs(x - ref_frame_idx))
                if closest_frame_idx in frame_idx_to_data:
                    key_frame_data.append(frame_idx_to_data[closest_frame_idx])
                else:
                    print(f"未找到帧索引 {closest_frame_idx} 的数据")
                    continue
            """
            if standard_info:
                # 使用 calculate_frame_sampling_for_video 来计算每个视频应采样的帧数
                frame_counts = calculate_frame_sampling_for_video(video_data, max_time_steps)

                # 将 frame_counts 和关键帧信息一起传递给 sample_frames_for_keyframes 进行采样
                sampled_frames = sample_frames_for_keyframes(video_data,
                                                         standard_info['standardSocreInfo']['firstKeyFrame'],
                                                         reference_frames,
                                                         fps,
                                                         max_time_steps,
                                                         frame_counts)

                # 根据采样的帧索引构建采样后的视频数据
                video_data = np.stack([video_data[idx] for idx in sampled_frames])

            if len(video_data) == 0:
                print(f"视频数据为空，跳过: {video_path}")
                continue

            video_data = np.stack(video_data)
            if augment:
                video_data = augment_data(video_data)
                video_data = augment_time(video_data, max_time_steps)
            else:
                video_data = align_and_normalize_time(video_data, max_time_steps)

            angles = calculate_angles(video_data)
            all_data.append(video_data)
            all_angles.append(angles)
            class_labels.append(action_number)

            # 构建评分文件路径并读取评分
            # 评分文件名与视频文件夹名一致，加上 .txt 后缀
            score_file_name = f"{video_folder}.txt"
            score_file_path = os.path.join(mark_dir, action_label, score_file_name)
            score = read_score(score_file_path)
            score_labels.append(score)
            key_labels.append(0)  # 如果没有关键动作标签，可以先设置为0

            video_id = video_folder  # 视频文件夹名作为视频唯一标识
            video_ids.append(video_id)

            print(f"已加载视频: {video_id}, 动作标签: {action_number}, 评分: {score}")

    print("数据加载完成")
    if len(all_data) == 0 or len(all_angles) == 0:
        raise ValueError("数据加载失败，all_data或all_angles为空，请检查数据目录和文件格式。")

    return all_data, all_angles, class_labels, score_labels, key_labels, video_ids

# 读取评分文件
def read_score(score_file_path):
    if os.path.exists(score_file_path):
        with open(score_file_path, 'r', encoding='utf-8') as f:
            score = f.read().strip()
            try:
                score = float(score)
                print(f"读取评分成功: {score_file_path} -> 评分: {score}")
            except ValueError:
                print(f"评分文件格式错误，无法转换为浮点数: {score_file_path}")
                score = 0.0
    else:
        score = 0.0  # 如果没有评分文件，默认评分为0
        print(f"未找到评分文件: {score_file_path}, 默认评分为0")
    return score

def save_processed_data(save_dir, all_data, all_angles, class_labels, score_labels, key_labels, video_ids):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = {
        'data': all_data,
        'angles': all_angles,
        'class_labels': class_labels,
        'score_labels': score_labels,
        'key_labels': key_labels,
        'video_ids': video_ids
    }
    save_path = os.path.join(save_dir, 'processed_data.pt')
    torch.save(dataset, save_path)
    print(f"处理后的数据已保存到 {save_path}")

def get_edge_index(edge_list):
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    reversed_edge_index = edge_index[[1, 0], :]
    edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    return edge_index
# 定义数据集类
class SkeletonDataset(Dataset):
    def __init__(self, data, angles, class_labels, score_labels=None, key_labels=None, video_ids=None):
        self.data = torch.tensor(data, dtype=torch.float)
        self.angles = torch.tensor(angles, dtype=torch.float)
        self.class_labels = torch.tensor(class_labels, dtype=torch.long)
        self.video_ids = video_ids
        if score_labels is not None:
            self.score_labels = torch.tensor(score_labels, dtype=torch.float)
        else:
            self.score_labels = None
        if key_labels is not None:
            self.key_labels = torch.tensor(key_labels, dtype=torch.float)
        else:
            self.key_labels = None

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        angle = self.angles[idx]
        Y_class = self.class_labels[idx]
        if self.score_labels is not None and self.key_labels is not None:
            Y_score = self.score_labels[idx]
            Y_key = self.key_labels[idx]
            if self.video_ids is not None:
                video_id = self.video_ids[idx]
                return X, angle, Y_class, Y_score, Y_key, video_id
            return X, angle, Y_class, Y_score, Y_key
        return X, angle, Y_class