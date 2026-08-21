import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
import mne
import numpy as np

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
event_dict = {'auditory/left': 1, 'visual/left': 3}

epochs = mne.Epochs(
    raw, events, event_id=event_dict,
    tmin=-0.5, tmax=1.0,  # 时间窗口拉长，方便看时频变化
    baseline=(-0.5, 0),
    preload=True,
    reject=dict(eeg=100e-6)
)

print(f"听觉左侧：{len(epochs['auditory/left'])} 个试次")
print(f"视觉左侧：{len(epochs['visual/left'])} 个试次")

# ============================================================
# 1. 小波变换：时频分析的核心工具
# ============================================================
print("\n" + "=" * 70)
print("1. 小波变换（Morlet小波）")
print("=" * 70)

# 设置频率范围：2-40Hz，间隔2Hz
freqs = np.arange(2, 41, 2)
print(f"分析的频率点：{freqs} Hz")

# 小波的周期数：频率越高，周期数越多（平衡时频分辨率）
n_cycles = freqs / 2.0  # 每个频率用 频率/2 个周期
print(f"各频率的小波周期数：{n_cycles}")

# ============================================================
# 2. 计算时频表示（TFR）—— 单条件
# ============================================================
print("\n" + "=" * 70)
print("2. 计算时频表示（TFR）—— 听觉刺激")
print("=" * 70)

# 计算听觉条件的时频表示
print("正在计算听觉刺激的时频表示...")
power_aud, itc_aud = mne.time_frequency.tfr_morlet(
    epochs['auditory/left'],
    freqs=freqs,
    n_cycles=n_cycles,
    return_itc=True,
    average=True,  # 对所有trial取平均
    n_jobs=1
)

print(f"计算完成")
print(f"Power形状：{power_aud.data.shape}（通道数 × 频率数 × 时间点数）")
print(f"ITC形状：{itc_aud.data.shape}")

# ============================================================
# 3. 绘制时频图（功率 ERSP）
# ============================================================
print("\n" + "=" * 70)
print("3. 时频图：功率变化（ERSP）")
print("=" * 70)

# 选一个通道画时频图（选中间的通道）
ch_idx = 30
ch_name = power_aud.ch_names[ch_idx]

fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

# 画功率时频图，用dB表示（相对于基线）
power_aud.plot(
    picks=[ch_idx],
    baseline=(-0.5, 0),
    mode='logratio',  # 用对数比表示，单位dB
    axes=ax,
    show=False,
    colorbar=True
)
ax.set_title(f'听觉左侧刺激 - 通道 {ch_name} - 功率时频图 (ERSP)', fontsize=14)
ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('频率 (Hz)', fontsize=12)
plt.tight_layout()
plt.show(block=True)

# ============================================================
# 4. 绘制时频图（相位一致性 ITC）
# ============================================================
print("\n" + "=" * 70)
print("4. 时频图：试次间相位一致性（ITC）")
print("=" * 70)

fig3, ax = plt.subplots(1, 1, figsize=(12, 6))

itc_aud.plot(
    picks=[ch_idx],
    axes=ax,
    show=False,
    colorbar=True
)
ax.set_title(f'听觉左侧刺激 - 通道 {ch_name} - 试次间相位一致性 (ITC)', fontsize=14)
ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('频率 (Hz)', fontsize=12)
plt.tight_layout()
plt.show(block=True)

# ============================================================
# 5. 条件对比：听觉 vs 视觉的时频差异
# ============================================================
print("\n" + "=" * 70)
print("5. 条件对比：听觉 vs 视觉的时频差异")
print("=" * 70)

# 计算视觉条件的时频表示
print("正在计算视觉刺激的时频表示...")
power_vis, itc_vis = mne.time_frequency.tfr_morlet(
    epochs['visual/left'],
    freqs=freqs,
    n_cycles=n_cycles,
    return_itc=True,
    average=True,
    n_jobs=1
)

# 计算差异（dB差）
power_diff = power_aud.copy()
power_diff.data = 10 * np.log10(power_aud.data / power_vis.data)  # 转成dB差

fig4, ax = plt.subplots(1, 1, figsize=(12, 6))

power_diff.plot(
    picks=[ch_idx],
    axes=ax,
    show=False,
    colorbar=True
)
ax.set_title(f'时频差异：听觉 - 视觉 (dB) - 通道 {ch_name}', fontsize=14)
ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('频率 (Hz)', fontsize=12)
plt.tight_layout()
plt.show(block=True)

