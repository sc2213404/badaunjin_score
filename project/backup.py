import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import ChebConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import random
import time
import csv
from model import STConv
from model import TemporalConv
from dataload import load_standard_info,load_skeleton_data_with_angles,read_score,augment_data,augment_time,align_and_normalize_time,calculate_angle,SkeletonDataset,save_processed_data

import torch
import torch.nn as nn


class STAttention(nn.Module):
    def __init__(self, hidden_channels):
        """
        初始化空间-时间注意力层，基于多头自注意力机制。

        参数：
            hidden_channels (int): 隐藏层通道数，即输入特征的维度。
        """
        super(STAttention, self).__init__()
        # 初始化多头自注意力机制，使用 4 个注意力头
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=4)

    def forward(self, X):
        """
        前向传播函数，应用多头自注意力机制。

        参数：
            X (torch.FloatTensor): 输入张量，形状为 (batch_size, time_steps, num_nodes, hidden_channels)。

        返回：
            attn_output (torch.FloatTensor): 自注意力输出，形状为 (batch_size, time_steps, num_nodes, hidden_channels)。
        """
        print("STAttention - 输入X的形状:", X.shape)

        # 提取维度信息
        batch_size, time_steps, num_nodes, hidden_channels = X.shape

        # 展平 time 和 nodes 维度，并使用 reshape 处理非连续内存张量
        X = X.permute(1, 0, 2, 3).reshape(time_steps, batch_size * num_nodes, hidden_channels)
        print("STAttention - 变换后X的形状:", X.shape)

        # 应用多头注意力机制
        attn_output, _ = self.attention(X, X, X)

        # 转换回原始形状
        attn_output = attn_output.permute(1, 0, 2).contiguous().reshape(batch_size, time_steps, num_nodes,
                                                                        hidden_channels)
        print("STAttention - 输出attn_output的形状:", attn_output.shape)

        return attn_output


# 定义STGCNMultiTask模型
class STGCNMultiTask(nn.Module):
    def __init__(self, num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers=3):
        super(STGCNMultiTask, self).__init__()
        self.st_convs = nn.ModuleList([
            STConv(
                num_nodes=num_nodes,
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                K=K,
            ) for i in range(num_layers)
        ])
        self.angle_fc = nn.Linear(num_angles, hidden_channels)
        self.attention = STAttention(hidden_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.key_action_detector = nn.Linear(hidden_channels, 1)
        self.regressor = nn.Linear(hidden_channels, num_angles)

    def forward(self, X, angles, edge_index, edge_weight=None):
        print("STGCNMultiTask - 输入X的形状:", X.shape)
        for st_conv in self.st_convs:
            X = st_conv(X, edge_index, edge_weight)
            X = F.relu(X)
        X = self.attention(X)
        X = X.mean(dim=2).mean(dim=1)
        angle_features = self.angle_fc(angles)
        X = X + angle_features  # 融合角度特征
        logits = self.classifier(X)
        key_action = torch.sigmoid(self.key_action_detector(X))
        angle_scores = self.regressor(X)
        print("STGCNMultiTask - 输出logits的形状:", logits.shape)
        return logits, key_action, angle_scores













# 定义边（根据关节点编号）
edge_list = [
    (15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8),
    (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10),
    (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14),
    (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)
]

def get_edge_index(edge_list):
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    reversed_edge_index = edge_index[[1, 0], :]
    edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
    return edge_index

edge_index = get_edge_index(edge_list)

# 设置数据路径和参数
data_dir = 'F:/软件工程/AI技术赛道（更新）/action'  # 请替换为您的数据集路径
mark_dir = 'F:/软件工程/AI技术赛道（更新）/add/mark_all'  # 评分信息所在的目录
max_time_steps = 100

# 定义保存路径
save_dir = 'F:/软件工程/AI技术赛道（更新）/processed_data'
dataset_path = os.path.join(save_dir, 'processed_data.pt')

# 检查是否已存在保存的数据
if os.path.exists(dataset_path):
    # 如果数据存在，加载已处理的数据
    dataset = torch.load(dataset_path)
    all_data = dataset['data']
    all_angles = dataset['angles']
    class_labels = dataset['class_labels']
    score_labels = dataset['score_labels']
    key_labels = dataset['key_labels']
    video_ids = dataset['video_ids']
    print("已加载保存的数据。")
else:
    # 如果数据不存在，加载和处理数据
    all_data, all_angles, class_labels, score_labels, key_labels, video_ids = load_skeleton_data_with_angles(
        data_dir, mark_dir, augment=True, max_time_steps=max_time_steps
    )

    # 保存处理后的数据
    os.makedirs(save_dir, exist_ok=True)
    save_processed_data(save_dir, all_data, all_angles, class_labels, score_labels, key_labels, video_ids)
    print("数据处理完成并已保存。")

# 输出数据维度和标签信息
print("数据形状 - all_data:", np.array(all_data).shape)
print("数据形状 - all_angles:", np.array(all_angles).shape)
print("标签 - class_labels:", class_labels)
print("评分 - score_labels:", score_labels)
print("视频ID - video_ids:", video_ids)


aligned_data = np.stack(all_data)  # (num_samples, max_time_steps, num_nodes, in_channels)
aligned_angles = np.stack(all_angles)  # (num_samples, num_time_steps, num_angles)

# 将角度特征降维（取平均或其他方法）
aligned_angles = aligned_angles.mean(axis=1)  # (num_samples, num_angles)

# 拆分数据集
train_data, val_data, train_angles, val_angles, train_class_labels, val_class_labels, train_score_labels, val_score_labels, train_key_labels, val_key_labels, train_video_ids, val_video_ids = train_test_split(
    aligned_data, aligned_angles, class_labels, score_labels, key_labels, video_ids, test_size=0.2, random_state=42,
    stratify=class_labels
)

# 创建数据集和数据加载器
train_dataset = SkeletonDataset(train_data, train_angles, train_class_labels, train_score_labels, train_key_labels, train_video_ids)
val_dataset = SkeletonDataset(val_data, val_angles, val_class_labels, val_score_labels, val_key_labels, val_video_ids)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用的设备:", device)

# 定义模型参数
num_nodes = 33
num_angles = 4  # 左臂弯、右臂弯、左腿弯、右腿弯
in_channels = 2  # 关键点的x和y坐标
hidden_channels = 64
num_classes = 15  # 动作类别数
kernel_size = 3
K = 3
num_layers = 3

# 初始化模型、损失函数和优化器
model = STGCNMultiTask(num_nodes, num_angles, in_channels, hidden_channels, num_classes, kernel_size, K, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义角度权重和误差等级配置
angle_weights = torch.tensor([0.1, 0.25, 0.10, 0.05], dtype=torch.float).to(device)  # 根据具体配置调整
tolerance_levels = [
    [10, 20, 30],  # 左臂弯
    [10, 20, 30],  # 右臂弯
    [10, 20, 30],  # 左腿弯
    [10, 20, 30],  # 右腿弯
]
score_levels = [
    [1, 0.8, 0.6, 0],  # 左臂弯
    [1, 0.8, 0.6, 0],  # 右臂弯
    [1, 0.8, 0.6, 0],  # 左腿弯
    [1, 0.8, 0.6, 0],  # 右腿弯
]

# 定义损失函数
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion_class = nn.CrossEntropyLoss(weight=class_weights)
criterion_key = nn.BCELoss()
criterion_score = nn.MSELoss()

# 定义训练函数
def compute_angle_score(user_angles, standard_angles, angle_weights, tolerance_levels, score_levels):
    batch_size, num_angles = user_angles.shape
    standard_angles = standard_angles.unsqueeze(0).repeat(batch_size, 1)
    angle_diff = torch.abs(user_angles - standard_angles)
    scores = torch.zeros_like(angle_diff)
    for i in range(num_angles):
        scores[:, i] = torch.where(
            angle_diff[:, i] <= tolerance_levels[i][0], score_levels[i][0],
            torch.where(
                angle_diff[:, i] <= tolerance_levels[i][1], score_levels[i][1],
                torch.where(
                    angle_diff[:, i] <= tolerance_levels[i][2], score_levels[i][2],
                    score_levels[i][3]
                )
            )
        )
    weighted_scores = scores * angle_weights
    total_score = weighted_scores.sum(dim=1) / angle_weights.sum()
    return total_score

def train(model, data_loader, criterion_class, criterion_key, criterion_score, optimizer, edge_index, device,
          standard_angles, angle_weights, tolerance_levels, score_levels):
    model.train()
    total_loss = 0
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
        optimizer.zero_grad()
        outputs_class, outputs_key, outputs_angle_scores = model(X, angles, edge_index.to(device))
        loss_class = criterion_class(outputs_class, Y_class)
        loss_key = criterion_key(outputs_key.squeeze(), Y_key.float())
        angle_score = compute_angle_score(outputs_angle_scores,
                                          torch.tensor(standard_angles, dtype=torch.float).to(device),
                                          angle_weights,
                                          tolerance_levels,
                                          score_levels)
        loss_score = criterion_score(angle_score, Y_score)
        loss = loss_class + loss_key + loss_score
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"训练损失: {avg_loss:.4f}")
    return avg_loss

# 定义测试函数
def test(model, data_loader, criterion_class, criterion_key, criterion_score, edge_index, device, standard_angles,
         angle_weights, tolerance_levels, score_levels):
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
            angle_score = compute_angle_score(outputs_angle_scores,
                                              torch.tensor(standard_angles, dtype=torch.float).to(device),
                                              angle_weights,
                                              tolerance_levels,
                                              score_levels)
            loss_score = criterion_score(angle_score, Y_score)
            loss = loss_class + loss_key + loss_score
            total_loss += loss.item()
            _, predicted = torch.max(outputs_class.data, 1)
            correct += (predicted == Y_class).sum().item()
            total_mae += F.l1_loss(angle_score.squeeze(), Y_score, reduction='sum').item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(Y_class.cpu().numpy())
    accuracy = correct / len(data_loader.dataset)
    mae = total_mae / len(data_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    avg_loss = total_loss / len(data_loader)
    print(f"验证损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, MAE: {mae:.4f}, F1分数: {f1:.4f}")
    return avg_loss, accuracy, mae, f1

# 训练模型
num_epochs = 50
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
best_val_acc = 0
patience = 10
counter = 0

train_losses = []
val_losses = []
val_accuracies = []
val_maes = []
val_f1s = []

standard_angles = [45, 90, 30, 90]  # 示例标准角度，实际根据动作定义

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, criterion_class, criterion_key, criterion_score, optimizer, edge_index,
                       device, standard_angles, angle_weights, tolerance_levels, score_levels)
    val_loss, val_accuracy, val_mae, val_f1 = test(model, val_loader, criterion_class, criterion_key, criterion_score,
                                                   edge_index, device, standard_angles, angle_weights, tolerance_levels,
                                                   score_levels)
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
        torch.save(model.state_dict(), 'best_model.pth')
        print("保存最佳模型")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("触发早停")
            break

# 测试并保存结果
def test_and_save_results(model, data_loader, edge_index, device, standard_angles, angle_weights, tolerance_levels, score_levels, output_csv='submit.csv'):
    model.eval()
    all_preds = []
    all_targets = []
    video_ids = data_loader.dataset.video_ids  # 假设数据集有video_ids属性
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
            Y_class = Y_class.to(device)
            start_time = time.time()
            outputs_class, outputs_key, outputs_angle_scores = model(X, angles, edge_index.to(device))
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            _, predicted = torch.max(outputs_class.data, 1)
            angle_score = compute_angle_score(outputs_angle_scores,
                                              torch.tensor(standard_angles, dtype=torch.float).to(device),
                                              angle_weights,
                                              tolerance_levels,
                                              score_levels)
            # 遍历batch中的每个样本，保存结果
            batch_size = X.size(0)
            for i in range(batch_size):
                video_id = batch_video_ids[i]
                action_label = predicted[i].item()
                score = angle_score[i].item()
                time_cost = inference_time / batch_size  # 平均到每个样本
                results.append([video_id, action_label, score, time_cost])
    # 保存CSV文件
    output_csv = '队长手机号_submit.csv'  # 请替换为您的队长手机号
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['视频文件唯一标识', '动作分类标签', '动作标准度评分', '推理总耗时(ms)'])
        csvwriter.writerows(results)
    print(f"结果已保存到 {output_csv}")

# 调用测试并保存结果的函数
test_and_save_results(model, val_loader, edge_index, device, standard_angles, angle_weights, tolerance_levels, score_levels)

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.legend()
plt.title('损失曲线')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='验证准确率')
plt.plot(val_maes, label='验证MAE')
plt.plot(val_f1s, label='验证F1分数')
plt.legend()
plt.title('验证指标')

plt.show()

# 保存最终模型
torch.save(model.state_dict(), 'final_model.pth')
print("模型已保存")
