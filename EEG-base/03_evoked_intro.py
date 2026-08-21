import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mne

# 全局中英文配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# 数字、坐标轴单位、英文符号专用字体（自带μ希腊字母）
plt.rcParams["font.serif"] = ["Times New Roman"]

# ========== 加载数据 + 分段 ==========
data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(data_file, preload=True)
raw.pick_types(eeg=True, stim=True)
raw.set_montage('standard_1020', match_case=False, on_missing='ignore')

events = mne.find_events(raw)
event_dict = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4
}

epochs = mne.Epochs(
    raw, events, event_id=event_dict,
    tmin=-0.2, tmax=0.5, baseline=(-0.2, 0),
    preload=True
)

# ========== 第一步：叠加平均生成 Evoked ==========
# 对每个条件的所有 trial 求平均
evoked_aud_left = epochs['auditory/left'].average()
evoked_vis_left = epochs['visual/left'].average()

print("Evoked 对象信息：")
print(evoked_aud_left)
print(f"\n数据形状：{evoked_aud_left.data.shape}")
print("含义：(通道数, 时间点数) — trial维度被平均掉了，变回2D")

# ========== 第二步：画ERP波形图（所有通道叠在一起）==========
print("\n绘制听觉左侧ERP波形...")
fig1 = evoked_aud_left.plot(time_unit='s')
fig1.set_size_inches(14, 8)
plt.show(block=False)


# ========== 第三步：两种条件对比 ==========
print("\n绘制听觉 vs 视觉对比...")
fig_list = mne.viz.plot_compare_evokeds(
    # 键=图例名称，值=对应evoked数据
    {'听觉左侧': evoked_aud_left, '视觉左侧': evoked_vis_left},
    picks='eeg',
    combine='mean',
    legend='upper right',
    show_sensors=False,
    show=False
)
fig2 = fig_list[0]
ax = fig2.axes[0]

# 1. 自动读取当前波形最大、最小电压值
y_min, y_max = ax.get_ylim()
# 2. 上下各拓宽 0.3μV 余量，保证线条不碰边框
new_ylim = (y_min - 0.3, y_max + 0.3)
ax.set_ylim(new_ylim)

# 绘制参考线
ax.axhline(0, c='black', lw=1.2)
ax.axvline(0, c='black', ls='--', lw=1.2)

plt.tight_layout()
plt.show(block=True)
