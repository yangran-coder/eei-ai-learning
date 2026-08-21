# eei-ai-learning
湖北大学计算机学院-计算机技术专硕-研究方向：脑电科学+深度学习
## 🧠关于这个仓库
记录我在脑电AI方向的学习过程，包括：
- 论文复现笔记
- 代码实现
- 实验记录和踩坑总结
## 📕数据文件说明
由于数据文件体积较大（超过 GitHub 100MB 限制），本项目不直接包含数据文件。
### 获取示例数据
使用 MNE 库自动下载：
```python
import mne

# 下载示例数据
data_path = mne.datasets.sample.data_path()

# 读取数据
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
```
## 💡项目结构
```
EEG-AI-Project/
├── .git/                          # Git 版本控制
├── .idea/                         # IDE 配置（已忽略）
├── EEG-base/                      # EEG数据处理基础代码
├── data/                          # 数据文件目录
│   └── MNE-sample-data/           # MNE 示例数据（需下载）
├── .gitignore                     # Git 忽略文件
├── LICENSE                        # 许可证
└── README.md                      # 项目说明
```

> ⚠️ **注意**: `MNE-sample-data/` 和 `*.fif` 文件因体积过大，已加入 `.gitignore`，请通过 MNE 库自动下载。
## 📄 许可证

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT © [2026] [yangran-coder](https://github.com/yangran-coder)

本项目采用 [MIT 许可证](LICENSE) 开源，你可以自由使用、修改和分发代码，只需保留版权声明。
