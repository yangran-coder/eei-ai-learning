import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# 0. 设置随机种子（保证结果可复现）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 1. 加载预处理好的数据
print("1. 加载数据")

data_path = r"D:\GitHub-Repositorys\EEG-AI\data\A01T_processed.npz"
data = np.load(data_path, allow_pickle=True)
X = data['X']  # (288, 22, 1000)
y = data['y']  # (288,)
ch_names = data['ch_names']

print(f"数据形状：{X.shape}")
print(f"标签形状：{y.shape}")
print(f"标签分布：{np.bincount(y)}")

# 2. 数据集划分（训练集 80% / 验证集 20%）
print("2. 数据集划分")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify保证各类别比例一致
)

print(f"训练集：{X_train.shape[0]} 个样本")
print(f"  标签分布：{np.bincount(y_train)}")
print(f"验证集：{X_val.shape[0]} 个样本")
print(f"  标签分布：{np.bincount(y_val)}")

# 3. 自定义Dataset和DataLoader
print("3. 创建数据加载器")


class EEGDataset(Dataset):
    """自定义EEG数据集"""

    def __init__(self, X, y):
        # X: (n_samples, n_channels, n_times) → 加一个维度变成 (n_samples, 1, n_channels, n_times)
        self.X = torch.FloatTensor(X).unsqueeze(1)  # 加channel维度
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 32

train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"批大小：{batch_size}")
print(f"训练集批次数：{len(train_loader)}")
print(f"验证集批次数：{len(val_loader)}")

# 4. 定义模型、损失函数、优化器
print("4. 定义模型、损失函数、优化器")

# 检测是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")


# 导入上一步定义的EEGNet（这里直接复制过来，方便独立运行）
class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_times=1000, n_classes=4,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        super(EEGNet, self).__init__()

        self.temporal_conv = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.temporal_bn = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(F1 * D)
        self.depthwise_pool = nn.AvgPool2d((1, 4))
        self.depthwise_dropout = nn.Dropout(dropout)

        self.separable_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.separable_bn = nn.BatchNorm2d(F2)
        self.separable_pool = nn.AvgPool2d((1, 8))
        self.separable_dropout = nn.Dropout(dropout)

        self.flatten_size = F2 * (n_times // 32)
        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        x = F.elu(self.temporal_bn(self.temporal_conv(x)))
        x = self.depthwise_dropout(self.depthwise_pool(F.elu(self.depthwise_bn(self.depthwise_conv(x)))))
        x = self.separable_pointwise(self.separable_depthwise(x))
        x = self.separable_dropout(self.separable_pool(F.elu(self.separable_bn(x))))
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


import torch.nn.functional as F

# 创建模型
model = EEGNet(n_channels=22, n_times=1000, n_classes=4).to(device)
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# 损失函数：交叉熵（多分类标准损失）
criterion = nn.CrossEntropyLoss()

# 优化器：Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器：验证集损失不下降时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

print("损失函数：CrossEntropyLoss")
print("优化器：Adam (lr=0.001)")
print("学习率调度：ReduceLROnPlateau（损失不降则减半）")

# 5. 训练循环
print("5. 开始训练")

n_epochs = 200
patience = 30  # 早停耐心值：验证集损失30个epoch不下降就停止

train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0
best_epoch = 0
epochs_no_improve = 0

for epoch in range(n_epochs):
    # ---------- 训练阶段 ----------
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # ---------- 验证阶段 ----------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # 验证不需要计算梯度
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # 记录历史
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # 学习率调度
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_eegnet.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # 每10个epoch打印一次
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}] "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc * 100:.1f}% | "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc * 100:.1f}%")

    # 早停
    if epochs_no_improve >= patience:
        print(f"\n早停触发！{patience}个epoch验证准确率没有提升")
        break

print(f"\n训练完成！")
print(f"最佳验证准确率：{best_val_acc * 100:.1f}% (第{best_epoch + 1}个epoch)")

# 6. 训练过程可视化
print("6. 训练过程可视化")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(train_losses, label='训练损失', linewidth=1.5)
axes[0].plot(val_losses, label='验证损失', linewidth=1.5)
axes[0].axvline(x=best_epoch, color='red', linestyle='--', label=f'最佳模型 (epoch {best_epoch + 1})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('损失')
axes[0].set_title('训练/验证损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(np.array(train_accs) * 100, label='训练准确率', linewidth=1.5)
axes[1].plot(np.array(val_accs) * 100, label='验证准确率', linewidth=1.5)
axes[1].axhline(y=25, color='gray', linestyle='--', label='机会水平 (25%)')
axes[1].axvline(x=best_epoch, color='red', linestyle='--', label=f'最佳模型 (epoch {best_epoch + 1})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('准确率 (%)')
axes[1].set_title('训练/验证准确率曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

# 7. 最终评估：混淆矩阵 + 分类报告
print("7. 最终评估")

# 加载最佳模型
model.load_state_dict(torch.load('best_eegnet.pth'))
model.eval()

# 在验证集上预测
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 分类报告
class_names = ['左手', '右手', '脚', '舌头']
print("\n分类报告：")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
ax.set_title(f'混淆矩阵 (验证集准确率: {best_val_acc * 100:.1f}%)', fontsize=14)
plt.tight_layout()
plt.show(block=True)

print("训练完成总结")
print(f"""
【结果】
  最佳验证准确率：{best_val_acc * 100:.1f}%
  机会水平（瞎猜）：25%
  超过机会水平：{(best_val_acc - 0.25) * 100:.1f}%
""")
