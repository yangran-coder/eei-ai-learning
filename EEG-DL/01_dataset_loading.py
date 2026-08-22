import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch

# 1. 从本地mat文件加载数据
print("1. 从本地mat文件加载BCI IV-2a数据 (A01T)")

mat_path = r"D:\GitHub-Repositorys\EEG-AI\data\MNE-bnci-data\database\data-sets\001-2014\A01T..mat"
mat = sio.loadmat(mat_path)
data = mat['data']  # (1, 9)

fs = 250  # 采样率
n_channels = 22  # 前22个是EEG通道
trial_len = 4  # 每个trial取4秒
n_samples = int(fs * trial_len)  # 1000个时间点

# BCI IV-2a的22个EEG通道名（标准10-20命名）
ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz']

all_trials = []
all_labels = []

# run 4-9 是有标签的训练数据（索引3-8）
for run_idx in range(3, 9):
    run = data[0, run_idx]
    X_run = run['X'][0, 0]  # (时间点, 25通道)
    trials_start = run['trial'][0, 0].flatten()  # trial起始样本点
    labels = run['y'][0, 0].flatten()  # 标签1-4

    print(f"  run {run_idx + 1}: {len(trials_start)} 个trial")

    for i, start in enumerate(trials_start):
        # 取从start开始的4秒数据，只取前22个EEG通道
        # X_run是(时间, 通道)，所以取 [start:start+n_samples, :22]
        trial = X_run[start:start + n_samples, :n_channels]
        # 转成(通道, 时间)格式
        trial = trial.T
        all_trials.append(trial)
        all_labels.append(labels[i] - 1)  # 标签转成0-3

X = np.array(all_trials)  # (288, 22, 1000)
y = np.array(all_labels)  # (288,)

print(f"\n加载完成")
print(f"数据形状：{X.shape} → (试次数, 通道数, 时间点数)")
print(f"标签形状：{y.shape}")
print(f"标签分布：{np.bincount(y)}")
print(f"  0=左手, 1=右手, 2=脚, 3=舌头")

# 2. 预处理：带通滤波 4-40Hz
print("2. 预处理：带通滤波 4-40Hz")


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """对每个通道做零相位带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # data形状: (n_trials, n_channels, n_times)
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[i, ch, :] = filtfilt(b, a, data[i, ch, :])
    return filtered


print("正在滤波...")
X_filtered = bandpass_filter(X, lowcut=4, highcut=40, fs=fs)
print("带通滤波完成 (4-40Hz)")

# 3. 预处理：标准化（每个被试每个通道单独标准化）
print("3. 标准化")

# 计算所有trial的均值和标准差（每个通道单独算）
mean = X_filtered.mean(axis=(0, 2), keepdims=True)  # (1, 22, 1)
std = X_filtered.std(axis=(0, 2), keepdims=True)  # (1, 22, 1)

X_normalized = (X_filtered - mean) / std

print(f"标准化前 - 均值: {X_filtered.mean():.4f}, 标准差: {X_filtered.std():.4f}")
print(f"标准化后 - 均值: {X_normalized.mean():.4f}, 标准差: {X_normalized.std():.4f}")
print("标准化完成")

# 4. 数据可视化：各类别平均波形
print("4. 可视化：各类别平均波形（C3/C4/Cz通道）")

class_names = ['左手', '右手', '脚', '舌头']
times = np.arange(n_samples) / fs  # 0-4秒

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
channels_to_plot = ['C3', 'Cz', 'C4']

for ax, ch_name in zip(axes, channels_to_plot):
    ch_idx = ch_names.index(ch_name)
    for i, name in enumerate(class_names):
        mask = y == i
        avg_wave = X_normalized[mask, ch_idx, :].mean(axis=0)
        ax.plot(times, avg_wave, linewidth=1.5, label=f'{name} (n={mask.sum()})')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('标准化幅值')
    ax.set_title(f'{ch_name}通道 - 各类别运动想象平均波形')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('时间 (s)')
plt.tight_layout()
plt.show(block=True)

# 5. 数据可视化：功率谱对比
print("\n" + "=" * 70)
print("5. 可视化：功率谱对比（C3/C4通道）")
print("=" * 70)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

for ax, ch_name in zip(axes2, ['C3', 'C4']):
    ch_idx = ch_names.index(ch_name)
    for i, name in enumerate(class_names):
        mask = y == i
        data_ch = X_filtered[mask, ch_idx, :]  # 用滤波后但未标准化的数据算PSD
        freqs, psd = welch(data_ch, fs=fs, nperseg=500, noverlap=250)
        ax.plot(freqs, psd.mean(axis=0), label=name, linewidth=1.5)

    ax.set_xlim([4, 40])
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('功率')
    ax.set_title(f'{ch_name}通道 - 各类别功率谱')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)

# 6. 保存处理好的数据（方便后面训练模型直接用）
print("6. 保存处理好的数据")

save_path = r"D:\GitHub-Repositorys\EEG-AI\data\A01T_processed.npz"
np.savez(save_path, X=X_normalized, y=y, ch_names=ch_names, fs=fs)
print(f"数据已保存到：{save_path}")
print(f"  X形状: {X_normalized.shape}")
print(f"  y形状: {y.shape}")

print("数据加载与预处理完成！")


