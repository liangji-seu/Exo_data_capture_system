为了让 Claude 帮你写出高质量的代码或架构设计，这个 Prompt 需要包含**背景上下文、数据特征、核心算法逻辑（借鉴论文）以及预期的技术栈**。

你可以直接复制下面的这段 Prompt 发给 Claude（建议使用 Claude 3.5 Sonnet，它在处理这类信号处理和深度学习任务时表现极佳）：

---

### 🚀 复制以下 Prompt 给 Claude

**Role:** 你是一位精通生物医学信号处理和深度学习的高级算法工程师，擅长处理超声（Ultrasound）或 EMG 信号。

**Background:**
我正在进行一项基于 A-mode 超声信号的手势预测/运动学追踪任务。目前面临的问题是：信号采样点多（1000点/帧）、噪声大、且存在极其严重的个体差异（Individual Variability），导致模型泛化能力差。我参考了 Nature Communications 的论文《Virtual reality interactions via a user-generic ultrasound human-machine interface》，希望借鉴其中的“中性参考（Neutral Referencing）”和“数据增强”策略来优化我的 4 通道超声数据模型。

**Data Specification:**
1. 输入：4 通道 A-mode 超声原始 RF 信号。
2. 规模：每一帧包含 4x1000 个数据点。
3. 目标：从超声回波的变化中预测手部动作/关节角度。

**Task: 请根据以下要求，为我提供 Python 代码实现方案（基于 PyTorch）：**

1. **信号预处理流水线（Preprocessing Pipeline）：**
   - 实现希尔伯特变换（Hilbert Transform）提取信号包络，以去除高频震荡，保留肌肉形变特征。
   - 提供一个可选的 STFT（短时傅里叶变换）函数，将 1D 信号转为 2D 时频图（Spectrogram）。

2. **中性参考策略（Neutral Referencing Strategy）：**
   - 实现一个数据加载机制：输入模型的不只是“当前时刻帧”，而是“当前帧”与“该用户静息态参考帧”的组合。
   - 请提供两种融合方式：(1) 直接相减（Current - Reference）；(2) 作为双通道输入（Channel Concatenation）。

3. **深度学习模型架构（Model Architecture）：**
   - 设计一个轻量级的 CNN-LSTM 模型。利用 1D/2D 卷积提取空间特征，利用 LSTM 处理动作的连续性。
   - 考虑 4 通道数据的融合特征，如何有效结合多通道信息。

4. **鲁棒性数据增强（Data Augmentation）：**
   - 参考论文，实现针对超声信号的增强算法：包括随机屏蔽部分采样点（Blank Portions）、模拟传感器偏移（Channel Shifting）以及添加高斯噪声。

5. **输出要求：**
   - 提供模块化的代码结构。
   - 解释为什么这种处理流程能解决“个体差异大”和“噪声敏感”的问题。

---

### 💡 为什么这个 Prompt 会有效？

1.  **明确了“中性参考”：** 这是解决你“因人而异”问题的关键，Claude 会重点处理 Reference Frame 的逻辑。
2.  **约束了预处理方法：** 提到 Hilbert Transform 和包络提取，能让模型避开原始 RF 信号的复杂震荡，直接学习肌肉的厚度变化。
3.  **模型选型精准：** CNN 处理单帧空间特征，LSTM 处理时间序列，这符合手势识别的物理本质。
4.  **借鉴前沿论文：** 直接点出那篇 Nature 论文，Claude 会在训练策略（如数据增强）上参考该论文的最优实践。

**你可以尝试发给它，如果它给出的代码太复杂，你可以继续追问：“请先帮我实现第一步：包络提取和中性参考的数据加载部分。”**