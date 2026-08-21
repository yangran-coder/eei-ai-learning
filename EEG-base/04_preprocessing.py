"""
EEG信号预处理完整流水线（MNE官方推荐顺序）
"""

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
import mne
import numpy as np

# 准备：加载数据
data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(data_file, preload=True)
raw.pick_types(eeg=True, eog=True, stim=True)  # 保留EEG、EOG、STIM通道

print(f"原始数据：{len(raw.ch_names)} 通道")
print(f"采样率：{raw.info['sfreq']:.1f} Hz")
print(f"时长：{raw.times[-1]:.1f} 秒")

# 保存一份原始数据用于最后对比
raw_original = raw.copy()

# ============================================================
# 第1步：检测并标记坏道（先标记，不插值）
# ============================================================
print("\n" + "=" * 70)
print("第1步：检测并标记坏道")
print("=" * 70)


# --- 检测方法：基于标准差的Z-score异常检测 ---
eeg_data = raw.copy().pick_types(eeg=True).get_data()
stds = np.std(eeg_data, axis=1)
mean_std = np.mean(stds)
std_of_std = np.std(stds)

print(f"所有EEG通道的平均标准差：{mean_std * 1e6:.2f} μV")

bads = []
eeg_ch_names = raw.copy().pick_types(eeg=True).ch_names
for i, ch_name in enumerate(eeg_ch_names):
    z = abs(stds[i] - mean_std) / std_of_std
    if z > 2.5:  # 偏离均值2.5个标准差以上算可疑
        bads.append(ch_name)
        print(f"  可疑坏道：{ch_name} (z={z:.2f})")

if len(bads) == 0:
    print("  自动检测未发现明显坏道")
    # 演示用：手动造一个坏道，方便后面看插值效果
    bads = ['EEG 010']
    print(f"  演示用：手动标记 {bads[0]} 为坏道")

# 标记到raw.info里
raw.info['bads'] = bads
print(f"\n已标记坏道：{raw.info['bads']}")

# ============================================================
# 第2步：带通滤波 + 陷波滤波
# ============================================================
print("\n" + "=" * 70)
print("第2步：带通滤波 + 陷波滤波")
print("=" * 70)

# 先画滤波前的PSD
fig1 = raw.compute_psd(fmin=0, fmax=100).plot(average=True)
fig1.suptitle('第2步前：滤波前PSD', fontsize=14)
plt.show(block=True)

# --- 带通滤波 ---
raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')
print("带通滤波完成：0.5 - 40 Hz")

# --- 陷波滤波 ---
# 去掉60Hz工频干扰（这个样本是美国的数据，所以是60Hz）
raw.notch_filter(freqs=60)
print("陷波滤波完成：60 Hz")

# 画滤波后的PSD
fig2 = raw.compute_psd(fmin=0, fmax=100).plot(average=True)
fig2.suptitle('第2步后：滤波后PSD (0.5-40Hz + 60Hz陷波)', fontsize=14)
plt.show(block=True)

# ============================================================
# 第3步：重参考（平均参考）
# ============================================================
print("\n" + "=" * 70)
print("第3步：重参考（平均参考）")
print("=" * 70)

raw.set_eeg_reference('average')
print("已设置为平均参考")

# 画重参考前后的对比（取第10个通道）
ch_idx = 10
fig3, axes = plt.subplots(2, 1, figsize=(14, 6))

# 重参考前的数据需要重新算（因为已经改了raw，所以用之前保存的备份）
raw_before_ref = raw_original.copy().pick_types(eeg=True)
raw_before_ref.filter(l_freq=0.5, h_freq=40)  # 同样滤波后再比
data_before = raw_before_ref.get_data()[ch_idx, :2000] * 1e6

data_after = raw.copy().pick_types(eeg=True).get_data()[ch_idx, :2000] * 1e6
times = raw.times[:2000]

axes[0].plot(times, data_before)
axes[0].set_title('重参考前（原始参考）')
axes[0].set_ylabel('幅值 (μV)')

axes[1].plot(times, data_after)
axes[1].set_title('重参考后（平均参考）')
axes[1].set_ylabel('幅值 (μV)')
axes[1].set_xlabel('时间 (s)')

plt.tight_layout()
plt.show(block=True)


# ============================================================
# 第4步：（可选）降采样
# ============================================================
print("\n" + "=" * 70)
print("第4步：降采样（可选）")
print("=" * 70)

print(f"降采样前采样率：{raw.info['sfreq']:.1f} Hz")
print(f"降采样前数据形状：{raw.get_data().shape}")

# 降到250Hz
raw.resample(sfreq=250)

print(f"降采样后采样率：{raw.info['sfreq']:.1f} Hz")
print(f"降采样后数据形状：{raw.get_data().shape}")
print("降采样完成")


# ============================================================
# 第5步：ICA拟合（最核心的一步）
# ============================================================
print("\n" + "=" * 70)
print("第5步：ICA拟合（核心步骤）")
print("=" * 70)

from mne.preprocessing import ICA

# 设置ICA参数
ica = ICA(
    n_components=15,  # 提取15个独立成分（一般15-20个足够）
    random_state=42,  # 固定随机种子，保证每次结果一样
    max_iter='auto',  # 迭代次数
    method='picard'
)

print("正在拟合ICA，请稍候...")
ica.fit(raw)
print("ICA拟合完成")

print(f"\n提取了 {ica.n_components_} 个独立成分")
explained_variance = ica.get_explained_variance_ratio(raw)
total_var = explained_variance["eeg"]
print(f"所有独立成分总共解释信号方差占比：{total_var * 100:.1f}%")

# --- 画ICA成分拓扑图 ---
print("\n绘制ICA成分拓扑图...")
fig4 = ica.plot_components()
plt.show(block=True)

# --- 自动识别眼电伪迹成分 ---
print("\n自动识别眼电伪迹成分...")
eog_idx, eog_scores = ica.find_bads_eog(raw)
print(f"检测到眼电相关成分：{eog_idx}")

# 画眼电成分的得分图
fig5 = ica.plot_scores(eog_scores, labels='EOG')
plt.show(block=True)


# ============================================================
# 第6步：应用ICA（剔除伪迹成分）
# ============================================================
print("\n" + "=" * 70)
print("第6步：应用ICA（剔除伪迹成分）")
print("=" * 70)

# 标记要剔除的成分
ica.exclude = eog_idx
print(f"将剔除以下伪迹成分：{ica.exclude}")

# 应用到raw数据
raw_clean = raw.copy()
ica.apply(raw_clean)
print("ICA伪迹剔除完成")

# --- 对比：ICA前后的波形 ---
print("\n绘制ICA前后对比...")
fig6, axes = plt.subplots(2, 1, figsize=(14, 8))

# 取前10秒，前5个通道
start, stop = 0, int(10 * raw.info['sfreq'])
ch_names = raw.copy().pick_types(eeg=True).ch_names[:5]

for i, ch in enumerate(ch_names):
    offset = i * 50  # 每个通道偏移50μV，方便看
    data_before = raw.copy().pick_types(eeg=True).get_data()[i, start:stop] * 1e6 + offset
    data_after = raw_clean.copy().pick_types(eeg=True).get_data()[i, start:stop] * 1e6 + offset
    times = raw.times[start:stop]

    axes[0].plot(times, data_before, label=ch, linewidth=0.8)
    axes[1].plot(times, data_after, label=ch, linewidth=0.8)

axes[0].set_title('ICA去伪迹前')
axes[0].set_ylabel('幅值 (μV)')
axes[0].legend(loc='upper right', fontsize=8)

axes[1].set_title('ICA去伪迹后')
axes[1].set_ylabel('幅值 (μV)')
axes[1].set_xlabel('时间 (s)')
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show(block=True)


# ============================================================
# 第7步：插值坏道
# ============================================================
print("\n" + "=" * 70)
print("第7步：插值坏道")
print("=" * 70)

print(f"插值前坏道：{raw_clean.info['bads']}")
print(f"插值前通道数：{len(raw_clean.ch_names)}")

raw_clean.interpolate_bads(reset_bads=True)
print("坏道插值完成")

print(f"插值后坏道：{raw_clean.info['bads']}")
print(f"插值后通道数：{len(raw_clean.ch_names)}")


# ============================================================
# 第8步：分段 Epoching
# ============================================================
print("\n" + "=" * 70)
print("第8步：分段 Epoching")
print("=" * 70)

# 找到事件
events = mne.find_events(raw_clean)
event_dict = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4
}

print(f"总事件数：{len(events)}")

# 分段：刺激前200ms 到 刺激后500ms
# 先不做基线校正，留到下一步
epochs = mne.Epochs(
    raw_clean,
    events,
    event_id=event_dict,
    tmin=-0.2,
    tmax=0.5,
    baseline=None,  # 先不做基线校正
    preload=True,
    reject=None  # 先不剔除，下一步做
)

print(f"\n分段完成：")
print(f"  总试次数：{len(epochs)}")
print(f"  数据形状：{epochs.get_data().shape}")
print(f"  含义：(试次数, 通道数, 时间点数)")


# ============================================================
# 第9步：基线校正
# ============================================================
print("\n" + "=" * 70)
print("第9步：基线校正")
print("=" * 70)

# 画基线校正前的平均波形
fig7, axes = plt.subplots(2, 1, figsize=(14, 8))

evoked_before = epochs['auditory/left'].average()
axes[0].plot(evoked_before.times, evoked_before.data.T * 1e6, linewidth=0.5, alpha=0.5)
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[0].set_title('基线校正前 - 听觉左侧刺激（所有通道）')
axes[0].set_ylabel('幅值 (μV)')

# 应用基线校正
epochs.apply_baseline(baseline=(-0.2, 0))
print("基线校正完成（基线期：-0.2 ~ 0 s）")

# 画基线校正后的平均波形
evoked_after = epochs['auditory/left'].average()
axes[1].plot(evoked_after.times, evoked_after.data.T * 1e6, linewidth=0.5, alpha=0.5)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[1].set_title('基线校正后 - 听觉左侧刺激（所有通道）')
axes[1].set_ylabel('幅值 (μV)')
axes[1].set_xlabel('时间 (s)')

plt.tight_layout()
plt.show(block=True)


# ============================================================
# 第10步：阈值剔除坏epoch
# ============================================================
print("\n" + "=" * 70)
print("第10步：阈值剔除坏epoch")
print("=" * 70)

print(f"剔除前总试次数：{len(epochs)}")

# 设置阈值：EEG通道超过100μV就扔掉
reject_criteria = dict(eeg=100e-6)  # 100 μV

epochs.drop_bad(reject=reject_criteria)
print(f"剔除后总试次数：{len(epochs)}")
print(f"被剔除的试次数：{len(epochs.drop_log)}")
print(f"各条件剩余试次：")
for cond in event_dict.keys():
    print(f"  {cond}: {len(epochs[cond])} 个")

# 画最终的平均ERP
print("\n绘制最终干净数据的平均ERP...")
fig8 = epochs['auditory/left'].average().plot(time_unit='s')
fig8.set_size_inches(14, 6)
fig8.suptitle('最终结果：听觉左侧刺激 ERP（预处理完成后）', fontsize=14)
plt.show(block=True)

# 保存最终结果
epochs.save('epochs_clean-epo.fif', overwrite=True)
print("\n预处理全部完成！干净的epochs已保存到 epochs_clean-epo.fif")
