import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import ChebConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import random
import time
import csv
from model import STConv,TemporalConv,STAttention,STGCNMultiTask
from dataload import load_standard_info, load_skeleton_data_with_angles, read_score, augment_data, augment_time, align_and_normalize_time, SkeletonDataset, save_processed_data,get_edge_index
from config import data_dir,mark_dir,max_time_steps,dataset_path,model_path
from config import num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers
from config import epochs ,best_val_acc,patience,counter
from assist import plot
import torch
import torch.nn as nn
# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight






# 定义边（根据关节点编号）
edge_list = [
    (15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8),
    (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10),
    (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14),
    (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)
]

edge_index = get_edge_index(edge_list)


# 检查是否已存在保存的数据
if os.path.exists(dataset_path):
    dataset = torch.load(dataset_path)
    all_data = dataset['data']
    all_angles = dataset['angles']
    class_labels = dataset['class_labels']
    score_labels = dataset['score_labels']
    key_labels = dataset['key_labels']
    video_ids = dataset['video_ids']
    print("已加载保存的数据。")
else:
    all_data, all_angles, class_labels, score_labels, key_labels, video_ids = load_skeleton_data_with_angles(
        data_dir, mark_dir, augment=True, max_time_steps=max_time_steps
    )
    os.makedirs(save_dir, exist_ok=True)
    save_processed_data(save_dir, all_data, all_angles, class_labels, score_labels, key_labels, video_ids)
    print("数据处理完成并已保存。")

aligned_data = np.stack(all_data)  # (num_samples, max_time_steps, num_nodes, in_channels)
aligned_angles = np.stack(all_angles)  # (num_samples, num_time_steps, num_angles)
aligned_angles = aligned_angles.mean(axis=1)  # (num_samples, num_angles)

# 拆分数据集
train_data, val_data, train_angles, val_angles, train_class_labels, val_class_labels, train_score_labels, val_score_labels, train_key_labels, val_key_labels, train_video_ids, val_video_ids = train_test_split(
    aligned_data, aligned_angles, class_labels, score_labels, key_labels, video_ids, test_size=0.2, random_state=42,
    stratify=class_labels
)

# 创建数据集和数据加载器
train_dataset = SkeletonDataset(train_data, train_angles, train_class_labels, train_score_labels, train_key_labels, train_video_ids)
val_dataset = SkeletonDataset(val_data, val_angles, val_class_labels, val_score_labels, val_key_labels, val_video_ids)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用的设备:", device)



# 初始化模型、损失函数和优化器
model = STGCNMultiTask(num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


"""
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("已加载训练好的模型。")
else:
    print("未找到已训练的模型，从头开始训练。")
"""  
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_acc = checkpoint['best_val_acc']
    print("已加载训练好的模型和优化器状态。")
else:
    print("未找到已训练的模型，从头开始训练。")





class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion_class = nn.CrossEntropyLoss(weight=class_weights)
criterion_key = nn.BCELoss()
criterion_score = nn.MSELoss()

# 训练函数
def train(model, data_loader, criterion_class, criterion_key, criterion_score, optimizer, edge_index, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        if len(batch) == 6:
            X, angles, Y_class, Y_score, Y_key, _ = batch
        else:
            X, angles, Y_class, Y_score, Y_key = batch
        X = X.to(device)
        angles = angles.to(device)

        Y_class = Y_class.to(device)
        Y_score = Y_score.to(device)
        Y_key = Y_key.to(device)
        optimizer.zero_grad()
        outputs_class, outputs_key, outputs_angle_scores = model(X, angles, edge_index.to(device))
        # 确保 Y_score 形状为 [batch_size]
        Y_score = Y_score.squeeze(-1)
        loss_class = criterion_class(outputs_class, Y_class)
        loss_key = criterion_key(outputs_key.squeeze(), Y_key.float())
        loss_score = criterion_score(outputs_angle_scores.squeeze(), Y_score)
        print(loss_class,loss_score)
        loss = loss_class+10*loss_score
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 仅在第一个批次打印
        if (batch_idx +1):
            print(f"Batch {batch_idx + 1}:")
            # 获取预测的类别标签
            _, predicted_classes = torch.max(outputs_class.data, 1)
            predicted_classes = predicted_classes.cpu().numpy()

            print("Predicted classes vs True classes:")
            for pred, true in zip(predicted_classes, Y_class.cpu().numpy()):
                print(f"Predicted: {pred}, True: {true}")

            # 打印预测角度得分与真实得分
            outputs_angle_scores_np = outputs_angle_scores.squeeze().detach().cpu().numpy()
            Y_score_np = Y_score.cpu().numpy()
            print("Predicted angle scores vs True scores:")
            for pred_score, true_score in zip(outputs_angle_scores_np, Y_score_np):
                print(f"Predicted score: {pred_score}, True score: {true_score}")
            print("-" * 50)
    avg_loss = total_loss / len(data_loader)
    print(f"训练损失: {avg_loss:.4f}")
    return avg_loss

# 测试函数
def test(model, data_loader, criterion_class, criterion_key, criterion_score, edge_index, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_mae = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 6:
                X, angles, Y_class, Y_score, Y_key, _ = batch
            else:
                X, angles, Y_class, Y_score, Y_key = batch
            X = X.to(device)
            angles = angles.to(device)
            Y_class = Y_class.to(device)
            Y_score = Y_score.to(device)
            Y_key = Y_key.to(device)
            outputs_class, outputs_key, outputs_angle_scores = model(X, angles, edge_index.to(device))
            loss_class = criterion_class(outputs_class, Y_class)
            loss_key = criterion_key(outputs_key.squeeze(), Y_key.float())
            loss_score = criterion_score(outputs_angle_scores.squeeze(), Y_score)
            loss = loss_class  +10* loss_score
            total_loss += loss.item()
            _, predicted = torch.max(outputs_class.data, 1)
            correct += (predicted == Y_class).sum().item()
            total_mae += F.l1_loss(outputs_angle_scores.squeeze(), Y_score, reduction='sum').item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(Y_class.cpu().numpy())
    accuracy = correct / len(data_loader.dataset)
    mae = total_mae / len(data_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    avg_loss = total_loss / len(data_loader)
    print(f"验证损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, MAE: {mae:.4f}, F1分数: {f1:.4f}")
    return avg_loss, accuracy, mae, f1
def predict(model, data_loader, edge_index, device):
    model.eval()  # 设置模型为评估模式
    predictions = []
    angle_scores = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 6:
                X, angles, _, _, _, _ = batch
            else:
                X, angles, _, _, _ = batch
            X = X.to(device)
            angles = angles.to(device)

            # 调用模型的 predict 方法
            predicted_classes, predicted_angle_scores = model.predict(X, angles, edge_index.to(device))

            # 将预测结果添加到列表中
            predictions.extend(predicted_classes.cpu().numpy())
            angle_scores.extend(predicted_angle_scores.cpu().numpy())

    return predictions, angle_scores

def test_and_save_results(model, data_loader, edge_index, device, output_csv='队长手机号_submit.csv'):
    model.eval()
    results = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                X, angles, Y_class, Y_score, Y_key, batch_video_ids = batch
            else:
                X, angles, Y_class, Y_score, Y_key = batch
                batch_video_ids = ["未知"] * X.size(0)
            X = X.to(device)
            angles = angles.to(device)

            # 开始计时
            start_time = time.time()
            # 模型推理
            outputs_class, outputs_key, outputs_angle_scores = model(X, angles, edge_index.to(device))
            # 计算推理时间
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            # 获取预测的动作分类标签
            _, predicted = torch.max(outputs_class.data, 1)
            # 获取预测的标准度评分
            angle_score = outputs_angle_scores.squeeze().cpu().numpy()
            # 遍历batch中的每个样本，保存结果
            batch_size = X.size(0)
            for i in range(batch_size):
                video_id = batch_video_ids[i]
                action_label = predicted[i].item()
                # 如果 angle_score 是一维的，需要取第 i 个元素
                if angle_score.ndim > 0:
                    score = angle_score[i].item()
                else:
                    score = angle_score.item()
                time_cost = inference_time / batch_size  # 平均到每个样本
                results.append([video_id, action_label, score, time_cost])
    # 保存CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['视频文件唯一标识', '动作分类标签', '动作标准度评分', '推理总耗时(ms)'])
        csvwriter.writerows(results)
    print(f"结果已保存到 {output_csv}")





# 训练和评估
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)


train_losses = []
val_losses = []
val_accuracies = []
val_maes = []
val_f1s = []





def main(args):
    # Initialize the model, dataset, and other parameters
    global best_val_acc  # 声明 best_val_acc 为全局变量
    global data_dir, mark_dir, max_time_steps, save_dir, dataset_path, model_path
    global num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers
    global epochs, patience, counter,model
  

    # Load dataset and create data loaders
    if args.mode == 'train':
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = train(model, train_loader, criterion_class, criterion_key, criterion_score, optimizer,
                               edge_index,
                               device)
            val_loss, val_accuracy, val_mae, val_f1 = test(model, val_loader, criterion_class, criterion_key,
                                                           criterion_score,
                                                           edge_index, device)

            scheduler.step(val_accuracy)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_maes.append(val_mae)
            val_f1s.append(val_f1)
            print(
                f'Epoch {epoch + 1}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}, 验证MAE: {val_mae:.4f}, 验证F1分数: {val_f1:.4f}')

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                # torch.save(model.state_dict(), 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, model_path)

                print("保存最佳模型")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("触发早停")
                    break

    elif args.mode == 'predict':
        print("开始预测")
        # 对验证集进行预测
        predictions, angle_scores = predict(model, val_loader, edge_index, device)

        # 输出预测结果
        for idx, (prediction, angle_score) in enumerate(zip(predictions, angle_scores)):
            print(f"样本 {idx + 1}: 预测类别 = {prediction}, 角度得分 = {angle_score:.4f}")

        # 如果需要，可以计算并输出评估指标
        print("\n分类报告:")
        print(classification_report(val_class_labels, predictions))

        # 调用测试并保存结果的函数
        test_and_save_results(model, val_loader, edge_index, device, output_csv='队长手机号_submit.csv')



# Command line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test a Model")
    parser.add_argument("--mode", type=str, choices=['train', 'predict'],default='train',
                        help="Specify 'train' or 'predict' mode")
    parser.add_argument("--video_directory", type=str, help="Directory containing video files for testing")
    parser.add_argument("--result_directory", type=str, help="Directory to save results")
    parser.add_argument("--phone_number", type=str, help="Phone number for result submission")
    args = parser.parse_args()

    main(args)

    plot(epochs,train_losses,val_losses,val_accuracies)