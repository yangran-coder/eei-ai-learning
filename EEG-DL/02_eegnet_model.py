import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# EEGNet 模型定义
class EEGNet(nn.Module):


    def __init__(self, n_channels=22, n_times=1000, n_classes=4,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        """
        参数说明:
            n_channels: EEG通道数 (BCI IV-2a是22)
            n_times: 每个trial的时间点数 (4秒×250Hz=1000)
            n_classes: 分类类别数 (4分类)
            F1: 时间卷积的滤波器数量（第一组频率滤波器）
            D: 深度卷积的深度乘数（每个频率滤波器学D个空间模式）
            F2: 可分离卷积的滤波器数量（通常 F1×D）
            kernel_length: 时间卷积核的长度（64个时间点≈0.256秒）
            dropout: dropout比例
        """
        super(EEGNet, self).__init__()

        # ========== 第1层：时间卷积 ==========
        # 卷积核形状: (1, kernel_length) → 只在时间维度卷积
        # 作用: 学习频率滤波器
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),  # 保持时间维度长度不变
            bias=False
        )
        self.temporal_bn = nn.BatchNorm2d(F1)

        # ========== 第2层：深度空间卷积 ==========
        # groups=F1 → 每个输入通道独立卷积（深度卷积）
        # 卷积核形状: (n_channels, 1) → 在通道维度卷积
        # 作用: 每个频率滤波器独立学习空间分布
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,  # 关键：深度卷积，每个通道独立
            bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(F1 * D)
        self.depthwise_pool = nn.AvgPool2d(kernel_size=(1, 4))  # 时间维度4倍下采样
        self.depthwise_dropout = nn.Dropout(dropout)

        # ========== 第3层：可分离卷积 ==========
        # 分为两步：深度卷积 + 逐点卷积(1×1)
        # 作用: 混合特征，减少参数量
        self.separable_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,  # 深度卷积
            bias=False
        )
        self.separable_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),  # 逐点卷积，混合通道
            bias=False
        )
        self.separable_bn = nn.BatchNorm2d(F2)
        self.separable_pool = nn.AvgPool2d(kernel_size=(1, 8))  # 再8倍下采样
        self.separable_dropout = nn.Dropout(dropout)

        # ========== 第4层：分类层 ==========
        # 计算全连接层的输入维度
        # 经过两次池化: n_times // 4 // 8 = n_times // 32
        # 空间维度已经变成1了（深度卷积把n_channels压成1）
        self.flatten_size = F2 * (n_times // 32)
        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        """
        前向传播
        x: (batch, 1, n_channels, n_times)
        """
        # 第1层：时间卷积
        x = self.temporal_conv(x)  # (batch, F1, n_channels, n_times)
        x = self.temporal_bn(x)
        x = F.elu(x)  # ELU激活函数

        # 第2层：深度空间卷积
        x = self.depthwise_conv(x)  # (batch, F1*D, 1, n_times)
        x = self.depthwise_bn(x)
        x = F.elu(x)
        x = self.depthwise_pool(x)  # (batch, F1*D, 1, n_times//4)
        x = self.depthwise_dropout(x)

        # 第3层：可分离卷积
        x = self.separable_depthwise(x)  # 深度卷积
        x = self.separable_pointwise(x)  # 逐点卷积
        x = self.separable_bn(x)
        x = F.elu(x)
        x = self.separable_pool(x)  # (batch, F2, 1, n_times//32)
        x = self.separable_dropout(x)

        # 第4层：分类
        x = x.flatten(start_dim=1)  # (batch, F2 * n_times//32)
        x = self.classifier(x)  # (batch, n_classes)

        return x


# 测试模型
print("EEGNet 模型测试")

# 创建模型
model = EEGNet(
    n_channels=22,
    n_times=1000,
    n_classes=4,
    F1=8,
    D=2,
    F2=16,
    kernel_length=64,
    dropout=0.25
)

print(f"\n模型结构：")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n参数量统计：")
print(f"  总参数量：{total_params:,}")
print(f"  可训练参数量：{trainable_params:,}")

# 测试前向传播
print(f"\n前向传播测试：")
batch_size = 4
dummy_input = torch.randn(batch_size, 1, 22, 1000)  # (batch, 1, 通道, 时间)
print(f"  输入形状：{dummy_input.shape}")

with torch.no_grad():
    output = model(dummy_input)
print(f"  输出形状：{output.shape}")
print(f"  输出（未归一化的logits）：")
print(f"  {output}")

# 用softmax转成概率
probs = F.softmax(output, dim=1)
print(f"\n  Softmax概率：")
print(f"  {probs}")
print(f"  每行和为：{probs.sum(dim=1)}")

print("\n模型测试通过！")

