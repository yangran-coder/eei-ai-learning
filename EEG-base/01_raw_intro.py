import mne
import matplotlib

matplotlib.use('Qt5Agg')  # 用Qt5后端，交互更流畅
import matplotlib.pyplot as plt

# 全局调小字体，这样能显示更多内容
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 5,
    'ytick.labelsize': 7
})

# ==================== 1. 下载并加载示例数据 ====================
# 第一次运行会自动下载MNE自带的样本数据（约1.5GB）
data_dir = r"D:\GitHub-Repositorys\EEG-AI\data"
print("正在加载样本数据...")
sample_data_folder = mne.datasets.sample.data_path(path=data_dir)
data_file = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

# 读取为 Raw 对象（preload=True 表示把数据读到内存里，方便后续处理）
raw = mne.io.read_raw_fif(data_file, preload=True)

print("\n 数据加载成功！")
print("=" * 60)

# ==================== 2. 只保留EEG通道 ====================
# pick_types 用来筛选通道：eeg=True 只留脑电，eog=True 留眼电（去伪迹用），stim=True 留事件标记
raw.pick_types(eeg=True, eog=True, stim=True)

print(f"\n筛选后通道总数：{len(raw.ch_names)}")
print(f"前15个通道名：{raw.ch_names[:15]}")

# ==================== 3. 查看数据维度 ====================
# get_data() 返回 numpy 数组，形状：(通道数, 时间点数)
data = raw.get_data()
print(f"\n数据形状：{data.shape}")
print(f"含义：{data.shape[0]} 个通道 × {data.shape[1]} 个时间点")

# 时间数组（单位：秒）
print(f"总时长：{raw.times[-1]:.2f} 秒")
print(f"采样率：{raw.info['sfreq']:.1f} Hz（每秒采集 {raw.info['sfreq']:.0f} 个点）")

# ==================== 4. 设置电极位置（画拓扑图必须有）====================
# standard_1020 就是国际标准的10-20电极放置系统
raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
print("\n 电极位置已设置（10-20系统）")

# ==================== 5. 交互式浏览原始信号 ====================
print("\n正在打开波形浏览器...")
# duration=5 表示一屏显示5秒
# n_channels=15 表示一屏显示15个通道
# 可以用鼠标滚轮上下滚动看更多通道，左右拖动平移时间
raw.plot(
    duration=5.0,
    n_channels=20,
    title='EEG原始信号波形',
    scalings='auto',  # 自动调整幅值缩放
    block=True
)

# ==================== 6. 画功率谱密度（PSD）====================
# PSD图是EEG最常用的分析图之一，看不同频率的能量分布
print("\n绘制功率谱密度图...")
psd = raw.compute_psd(fmin=0.5, fmax=40)
psd.plot(average=True)

plt.show(block=True)  # 关掉PSD窗口后程序才结束