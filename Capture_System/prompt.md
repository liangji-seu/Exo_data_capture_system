# Role: 超声信号处理专家 (Ultrasound Signal Processing Expert)

## Task: 为 A-mode 超声数据实现基于 Nature 论文标准的预处理管线

### Context:
我正在进行一项基于 A-mode 超声预测肌肉力矩的研究。我有一组原始的射频（RF）采样数据。
**我的数据参数如下：**
- **帧率 (Frame Rate):** 100 Hz (两帧间隔 10ms)。
- **单帧采样点 (Samples per frame):** 1000 个点。
- **系统中心频率:** 5 MHz。

我需要你参考 Nature Communications (Jin et al., 2024) 的方法，编写一个 Python 类来实现预处理，目标是生成高清晰度的 M-mode 图像，用于后续的大腿股直肌厚度追踪。

### Processing Pipeline (请按以下步骤严格执行):

1. **带通滤波 (Band-pass Filtering):**
   - 实现一个 4–6 MHz 的 Butterworth 带通滤波器。
   - **作用：** 消除低频基线漂移和高频电子噪声，保留肌肉组织的反射特征。

2. **包络提取 (Envelope Detection):**
   - 使用 **希尔伯特变换 (Hilbert Transform)** 提取信号包络。
   - **关键要求：** 将振荡的 RF 信号转变为反映反射强度的解析信号。这是解决目前图像模糊、边界不清晰的核心步骤。

3. **对数压缩与时间增益补偿 (Log Compression & TGC):**
   - **对数压缩：** 使用 $20 \cdot \log_{10}$ 压缩动态范围，使弱信号可见。
   - **TGC：** 考虑到超声随深度衰减，请在算法中加入一个可调的线性增益补偿，增强 1000 个采样点中后半段（深层）的信号强度。

4. **2D 空间-时间联合平滑 (2D Smoothing):**
   - **深度方向 (Axis 0):** 使用窗口大小约 2mm 的高斯滤波器平滑斑点噪声。
   - **时间方向 (Axis 1):** 在相邻帧（例如 5 帧/50ms 窗口）间进行平滑，确保肌肉边界运动的连贯性。

5. **M-mode 可视化映射:**
   - 输入：`(1000, num_frames)` 的矩阵。
   - 输出：一张高质量的 2D 灰度/热力图。
   - 坐标转换：纵轴需根据声速（1540 m/s）将 1000 个点转换为深度（mm）。

### Technical Requirements:
- 使用 Python (NumPy, SciPy, Matplotlib)。
- 代码要求模块化：创建一个 `UltrasoundProcessor` 类，包含 `preprocess()` 和 `plot_m_mode()` 方法。
- 请在代码中通过注释解释每一步处理对“识别肌肉厚度”的物理贡献。
- 考虑到我一帧只有 1000 个点，请在计算深度坐标时务必准确。

### Output:
请提供完整的 Python 代码，并包含一个简单的生成模拟数据并绘图展示效果的示例。