import os
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import STConv,TemporalConv,STAttention,STGCNMultiTask
from dataload import load_standard_info, load_skeleton_data_with_angles, read_score, augment_data, augment_time, align_and_normalize_time, SkeletonDataset, save_processed_data,get_edge_index
from config import data_dir,mark_dir,max_time_steps,save_dir,dataset_path,model_path
from config import num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers
from config import epochs ,best_val_acc,patience,counter
from MediaPipe import process_all_videos_in_folder
# 加载预训练模型的函数
def load_model():
    model = STGCNMultiTask(num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model


# 处理视频并识别动作分类及其标准度分数的函数
def recognize_actions_and_scores_in_video(model, video_path):
    process_all_videos_in_folder(video_path,video_path)
    # 动作分类与评分的细节代码，参赛选手实现
    start_time = time.time()
    label = model.predict(video_path)
    score = model.predict(video_path)
    end_time = time.time()
    cost_time = (end_time - start_time) * 1000
    actions_scores_times = [
        {"label": label, "score": score, "times": cost_time},
    ]
    # 动作分类与评分的细节代码，参赛选手实现
    return actions_scores_times


# 将结果写入CSV文件的函数
def write_results_to_csv(results, output_csv):
    data = []
    for video_name, actions_and_scores in results.items():
        for entry in actions_and_scores:
            data.append([video_name, entry["label"], entry["score"], entry["times"]])

    df = pd.DataFrame(data, columns=["视频名称", "动作分类", "标注度评分", "推理耗时"])
    df.to_csv(output_csv, index=False)


# 主函数
def main(args):
    video_directory = args.video_directory  # 视频文件目录
    output_csv = os.path.join(args.result_directory, f"{args.phone_number}_submit.csv")  # 输出的CSV文件路径

    # 加载模型
    model = load_model()

    # 初始化一个字典来存储结果
    results = {}

    # 遍历视频目录中的文件
    for filename in os.listdir(video_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_directory, filename)
            # 识别视频中的动作及其分数
            actions_scores_times = recognize_actions_and_scores_in_video(model, video_path)
            # 存储识别结果
            results[filename] = actions_scores_times

    # 将结果写入CSV文件
    write_results_to_csv(results, output_csv)


# 设置命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频动作识别和打分")
    parser.add_argument("--video_directory", type=str, required=True, help="视频文件目录")
    parser.add_argument("--result_directory", type=str, required=True, help="结果输出目录")
    parser.add_argument("--phone_number", type=str, required=True, help="队长手机号")
    args = parser.parse_args()

    main(args)
