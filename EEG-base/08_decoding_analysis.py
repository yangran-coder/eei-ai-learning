import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
import mne
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ============================================================
# 准备：加载数据 + 预处理 + 分段
# ============================================================
print("=" * 70)
print("准备：加载数据 + 预处理")
print("=" * 70)

data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(data_file, preload=True)
raw.pick_types(eeg=True, stim=True)
raw.filter(l_freq=0.5, h_freq=40)
raw.set_eeg_reference('average')
raw.resample(250)

events = mne.find_events(raw)
# 二分类：听觉左 vs 视觉左
event_dict = {'auditory/left': 1, 'visual/left': 3}

epochs = mne.Epochs(
    raw, events, event_id=event_dict,
    tmin=-0.2, tmax=0.8,
    baseline=(-0.2, 0),
    preload=True,
    reject=dict(eeg=100e-6)
)

print(f"听觉左侧：{len(epochs['auditory/left'])} 个试次")
print(f"视觉左侧：{len(epochs['visual/left'])} 个试次")

# ============================================================
# 1. 数据准备：X 和 y
# ============================================================
print("\n" + "=" * 70)
print("1. 数据准备")
print("=" * 70)

# X = 特征矩阵 (n_samples, n_features)
# y = 标签 (n_samples,)
X = epochs.get_data()  # (n_trials, n_channels, n_times)
y = epochs.events[:, -1]  # 最后一列是事件ID

print(f"原始数据形状：{X.shape}")
print(f"  → (试次数, 通道数, 时间点数)")
print(f"标签：{np.unique(y)}")
print(f"  1 = 听觉左侧")
print(f"  3 = 视觉左侧")

# 传统机器学习需要把数据展平成2D：(n_trials, n_channels × n_times)
# 但先做"时间解码"——每个时间点单独训练，所以先不展平
print(f"\n时间解码：每个时间点单独训练分类器")
print(f"时间点数：{X.shape[2]}")
print(f"每个时间点的特征数：{X.shape[1]}（通道数）")

# ============================================================
# 2. 时间解码（Temporal Decoding）
# ============================================================
print("\n" + "=" * 70)
print("2. 时间解码：每个时间点的解码准确率")
print("=" * 70)

# 分类器：SVM（支持向量机），线性核
# 前面加StandardScaler做标准化，这是SVM必须的
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1.0, random_state=42)
)

# 交叉验证：5折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_times = X.shape[2]
scores = np.zeros(n_times)

print("正在训练分类器，逐个时间点解码...")
for t in range(n_times):
    # 取出第t个时间点的数据：(n_trials, n_channels)
    X_t = X[:, :, t]
    # 5折交叉验证
    score = cross_val_score(clf, X_t, y, cv=cv, scoring='accuracy')
    scores[t] = score.mean()

print("解码完成！")

# 计算机会水平（chance level）
chance = 1 / len(np.unique(y))
print(f"\n机会水平（瞎猜）：{chance * 100:.1f}%")
print(f"最高准确率：{scores.max() * 100:.1f}%")
print(f"最高准确率出现时间：{epochs.times[np.argmax(scores)] * 1000:.0f} ms")

# ============================================================
# 3. 可视化：时间解码曲线
# ============================================================
print("\n绘制时间解码曲线...")

fig1, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(epochs.times, scores * 100, linewidth=2, color='blue', label='SVM解码准确率')
ax.axhline(y=chance * 100, color='red', linestyle='--', linewidth=2, label=f'机会水平 ({chance * 100:.0f}%)')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.axvspan(0, 0, alpha=0)  # 占位

# 标记超过机会水平的区域
above_chance = scores > chance
if np.any(above_chance):
    # 找出连续的区间
    diff = np.diff(above_chance.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if above_chance[0]:
        starts = np.insert(starts, 0, 0)
    if above_chance[-1]:
        ends = np.append(ends, len(scores) - 1)

    for s, e in zip(starts, ends):
        ax.axvspan(epochs.times[s], epochs.times[e],
                   alpha=0.2, color='blue')

ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('解码准确率 (%)', fontsize=12)
ax.set_title('时间解码曲线：听觉 vs 视觉（SVM分类器）', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim([40, 100])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)


# ============================================================
# 4. 不同分类器对比
# ============================================================
print("\n" + "=" * 70)
print("4. 不同分类器对比：SVM vs LDA")
print("=" * 70)

# LDA分类器
clf_lda = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis()
)

scores_lda = np.zeros(n_times)
print("正在训练LDA分类器...")
for t in range(n_times):
    X_t = X[:, :, t]
    score = cross_val_score(clf_lda, X_t, y, cv=cv, scoring='accuracy')
    scores_lda[t] = score.mean()

print("LDA解码完成！")
print(f"LDA最高准确率：{scores_lda.max() * 100:.1f}%")

# 对比图
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(epochs.times, scores * 100, linewidth=2, color='blue', label='SVM')
ax.plot(epochs.times, scores_lda * 100, linewidth=2, color='orange', label='LDA')
ax.axhline(y=chance * 100, color='red', linestyle='--', linewidth=2, label=f'机会水平 ({chance * 100:.0f}%)')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('解码准确率 (%)', fontsize=12)
ax.set_title('分类器对比：SVM vs LDA', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim([40, 100])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)


print("\n解码分析部分完成！")
