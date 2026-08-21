import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import mne
import numpy as np

print("=" * 70)
print("准备：加载数据 + 预处理")
print("=" * 70)

data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(data_file, preload=True)

eeg_ch_names = [ch for ch in raw.ch_names if ch.startswith("EEG")]
raw.pick_channels(eeg_ch_names + ['STI 014'])  # 只保留EEG + 刺激通
raw.filter(l_freq=0.5, h_freq=40)
raw.set_eeg_reference('average')
raw.resample(250)

# 先提取事件（必须保留STI通道才能找trigger！）
events = mne.find_events(raw, stim_channel="STI 014")

# =====关键一步：提取事件完成后，移除所有stim触发通道=====
raw.pick_types(eeg=True, meg=False, eog=True, stim=False)
event_dict = {'auditory/left': 1, 'visual/left': 3}

epochs = mne.Epochs(
    raw, events, event_id=event_dict,
    tmin=-0.2, tmax=0.8,
    baseline=(-0.2, 0),
    preload=True,
    reject=dict(eeg=100e-6)  # EEG ±100μV阈值
)

print(f"听觉左侧：{len(epochs['auditory/left'])} 个试次")
print(f"视觉左侧：{len(epochs['visual/left'])} 个试次")

# ============================================================
# 1. 假差异演示
# ============================================================
print("\n" + "=" * 70)
print("1. 为什么需要统计检验？")
print("=" * 70)

aud_epochs = epochs['auditory/left'].get_data()
n_trials = len(aud_epochs)
half = n_trials // 2

np.random.seed(42)
indices = np.random.permutation(n_trials)
group1 = aud_epochs[indices[:half]].mean(axis=0)
group2 = aud_epochs[indices[half:]].mean(axis=0)

fake_diff = (group1 - group2) * 1e6

fig1, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(epochs.times, fake_diff.mean(axis=0), linewidth=2, color='purple')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('幅值差 (μV)', fontsize=12)
ax.set_title('假差异：同一条件随机分成两组的差异波（其实没有真差异）', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)

# ============================================================
# 2. 聚类置换检验
# ============================================================
print("\n" + "=" * 70)
print("2. 聚类置换检验（cluster-based permutation test）")
print("=" * 70)

X_aud = epochs['auditory/left'].get_data()
X_vis = epochs['visual/left'].get_data()
X = [X_aud, X_vis]

print(f"听觉数据形状：{X_aud.shape}")
print(f"视觉数据形状：{X_vis.shape}")

print("\n正在执行聚类置换检验，请稍候...")
threshold = 2.0
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
    X,
    threshold=threshold,
    n_permutations=1000,
    tail=0,
    n_jobs=1,
    seed=42
)

print(f"\n检验完成！")
print(f"找到 {len(clusters)} 个候选聚类")
print(f"其中显著的（p < 0.05）有 {sum(cluster_p_values < 0.05)} 个")

significant_clusters = np.where(cluster_p_values < 0.05)[0]
print(f"\n显著聚类详情：")
for i, idx in enumerate(significant_clusters):
    cluster = clusters[idx]
    p_val = cluster_p_values[idx]
    n_chans = len(np.unique(cluster[0]))
    n_times = len(np.unique(cluster[1]))
    print(f"  聚类{i + 1}：p = {p_val:.4f}，涉及 {n_chans} 个通道，{n_times} 个时间点")

# ============================================================
# 3. 可视化差异波
# ============================================================
print("\n" + "=" * 70)
print("3. 可视化：显著差异的时间窗口")
print("=" * 70)

diff_wave = (X_aud.mean(axis=0) - X_vis.mean(axis=0)) * 1e6
diff_mean = diff_wave.mean(axis=0)

fig2, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.plot(epochs.times, diff_mean, linewidth=2, color='black', label='差异波（听觉-视觉）')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

for idx in significant_clusters:
    cluster = clusters[idx]
    time_indices = np.unique(cluster[1])
    if len(time_indices) > 0:
        t_start = epochs.times[time_indices[0]]
        t_end = epochs.times[time_indices[-1]]
        ax.axvspan(t_start, t_end, alpha=0.3, color='red', label=f'显著 (p={cluster_p_values[idx]:.4f})')

ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('幅值差 (μV)', fontsize=12)
ax.set_title('听觉 vs 视觉 — 差异波与显著区间', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=True)
