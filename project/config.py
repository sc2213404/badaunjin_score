import os
# 设置数据路径和参数
data_dir = 'F:/软件工程/AI技术赛道（更新）/action'
mark_dir = 'F:/软件工程/AI技术赛道（更新）/add/mark_all'
max_time_steps = 100

# 定义保存路径

dataset_path = '../processed_data.pt'

model_path = 'best_model.pth'


# 定义模型参数
num_nodes = 33
num_angles = 8
in_channels = 2  # 关键点的x和y坐标
hidden_channels = 64
num_classes = 15  # 动作类别数
kernel_size = 3
K = 3
num_layers = 3
# 训练和评估
epochs = 150
best_val_acc = 0
patience = 30
counter = 0