import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
# 解决负号显示异常
plt.rcParams["axes.unicode_minus"] = False
import mne
import numpy as np

# ============================================================
# 准备：加载数据 + 快速预处理
# ============================================================
print("=" * 70)
print("准备：加载数据 + 快速预处理")
print("=" * 70)

data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(data_file, preload=True)
raw.pick_types(eeg=True, stim=True)

# 快速预处理
raw.filter(l_freq=0.5, h_freq=40)
raw.set_eeg_reference('average')
raw.resample(250)

# 分段
events = mne.find_events(raw)
event_dict = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4
}

epochs = mne.Epochs(
    raw, events, event_id=event_dict,
    tmin=-0.2, tmax=0.8,  # 时间窗口拉长一点，方便看成分
    baseline=(-0.2, 0),
    preload=True,
    reject=dict(eeg=100e-6)
)

print(f"总试次数：{len(epochs)}")
print(f"各条件试次数：")
for cond in event_dict:
    print(f"  {cond}: {len(epochs[cond])}")

# ============================================================
# 1. 生成各条件的 Evoked（叠加平均）
# ============================================================
print("\n" + "=" * 70)
print("1. 生成各条件的 Evoked（叠加平均）")
print("=" * 70)

evoked_aud_left = epochs['auditory/left'].average()
evoked_aud_right = epochs['auditory/right'].average()
evoked_vis_left = epochs['visual/left'].average()
evoked_vis_right = epochs['visual/right'].average()

print("4个条件的Evoked已生成")
print(f"每个Evoked的形状：{evoked_aud_left.data.shape}（通道数 × 时间点数）")

# ============================================================
# 2. 单条件 ERP 波形图（所有通道）
# ============================================================
print("\n" + "=" * 70)
print("2. 单条件 ERP 波形图（所有通道）")
print("=" * 70)

fig1 = evoked_aud_left.plot(time_unit='s')
fig1.set_size_inches(14, 8)
fig1.suptitle('听觉左侧刺激 - 所有通道ERP', fontsize=14)
plt.show(block=True)

# ============================================================
# 3. 单通道 ERP 对比（听觉 vs 视觉）
# ============================================================
print("\n" + "=" * 70)
print("3. 单通道 ERP 对比（听觉 vs 视觉）")
print("=" * 70)

# 选一个通道（这个样本数据通道名是编号的，选中间的）
# 通常听觉在Cz附近，视觉在Oz附近
ch_name = evoked_aud_left.ch_names[30]  # 选第31个通道（大概在中央区）

# 提取数据
aud_data = evoked_aud_left.data[evoked_aud_left.ch_names.index(ch_name)] * 1e6
vis_data = evoked_vis_left.data[evoked_vis_left.ch_names.index(ch_name)] * 1e6
times = evoked_aud_left.times

fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(times, aud_data, label='听觉左侧', linewidth=2, color='blue')
ax.plot(times, vis_data, label='视觉左侧', linewidth=2, color='red')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('时间 (s)', fontsize=12)
ax.set_ylabel('幅值 (μV)', fontsize=12)
ax.set_title(f'通道 {ch_name}：听觉 vs 视觉 ERP对比', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)


# ============================================================
# 4. 多条件对比图（蝴蝶图 + 拓扑图）
# ============================================================
print("\n" + "=" * 70)
print("4. 多条件联合对比（蝴蝶图）")
print("=" * 70)

# 听觉左 vs 视觉左 的对比
# 将两个 Evoked 对象放入字典，键名作为图例
evoked_dict = {
    '听觉左侧': evoked_aud_left,
    '视觉左侧': evoked_vis_left
}

fig3 = mne.viz.plot_compare_evokeds(
    evoked_dict,  # 直接传入字典，key 会自动用作 legend 标签
    picks='eeg',
    combine='mean',
    show_sensors=False,
    title='听觉左侧 vs 视觉左侧 - 全通道平均ERP'
)
plt.show(block=True)


# ============================================================
# 5. 差异波（条件相减）
# ============================================================
print("\n" + "=" * 70)
print("5. 差异波（条件相减）")
print("=" * 70)

# 计算差异波：听觉左 - 视觉左
evoked_diff = mne.combine_evoked(
    [evoked_aud_left, evoked_vis_left],
    weights=[1, -1]  # 1×听觉 + (-1)×视觉 = 听觉 - 视觉
)

fig4 = evoked_diff.plot(time_unit='s')
fig4.set_size_inches(14, 8)
fig4.suptitle('差异波：听觉左侧 - 视觉左侧', fontsize=14)
plt.show(block=True)


print("\nERP波形分析部分完成！")
