import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mne

# 全局调小字体，这样能显示更多内容
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 5,
    'ytick.labelsize': 7
})

# ========== 加载数据 ==========
data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(data_file, preload=True)
raw.pick_types(eeg=True, stim=True)
raw.set_montage('standard_1020', match_case=False, on_missing='ignore')

print("数据加载完成")
print("=" * 60)

# ========== 第一步：找到事件标记 ==========
# STIM通道记录了实验中什么时候出现了什么刺激
events = mne.find_events(raw)

print(f"\n总事件数量：{len(events)}")
print("\n前10个事件的格式：[时间点编号, 前一事件ID, 当前事件ID]")
print(events[:10])

# 这个实验的4种事件：
# 1 = 左耳听觉刺激
# 2 = 右耳听觉刺激
# 3 = 左侧视觉刺激
# 4 = 右侧视觉刺激
event_dict = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4
}

# 画事件分布图
fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'])
plt.show(block=False)

# ========== 第二步：分段（Epoching）==========
print("\n正在分段...")

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.2,       # 从刺激前200ms开始取
    tmax=0.5,        # 到刺激后500ms结束
    baseline=(-0.2, 0),  # 用刺激前200ms做基线校正
    preload=True,
    reject=dict(eeg=150e-6)  # 自动剔除幅值超过150微伏的坏段
)

print("\n" + "=" * 60)
print("Epochs 对象信息：")
print(epochs)
print(f"\n数据形状：{epochs.get_data().shape}")
print("含义：(试次数量, 通道数, 每个试次的时间点数)")
print(f"每个试次时长：{epochs.times[-1]:.3f} 秒")

# ========== 第三步：按条件筛选 ==========
# 只看"左侧视觉"的试次
epochs_vis_left = epochs['visual/left']
print(f"\n左侧视觉刺激的试次数：{len(epochs_vis_left)}")

# 所有听觉的（左+右合并）
epochs_aud = epochs['auditory']
print(f"所有听觉刺激的试次数：{len(epochs_aud)}")

# ========== 第四步：画Epochs图 ==========
print("\n打开Epochs浏览窗口...")
fig2 = epochs['visual/left'].plot(n_epochs=10, n_channels=15, block=True)
fig2.set_size_inches(16, 10)

# ========== 第五步：画ERP图像（热力图形式）==========
print("\n绘制所有试次的平均ERP图...")
fig3 = epochs['auditory/left'].plot_image(picks='eeg', combine='mean')
plt.show(block=True)
