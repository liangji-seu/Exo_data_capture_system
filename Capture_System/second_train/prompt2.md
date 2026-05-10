3. 为你准备的 Prompt（复制给 AI）
Prompt：请按照以下需求修改现有的步态预测模型代码（model.py 和 data_loader.py）

1. 数据加载优化 (data_loader.py):

单通道拆分： 修改 GaitDataset。原本 4 通道超声是一个样本，现在请将每个通道拆分为独立样本。即：输入从 (W, 4, 300) 变为 (W, 1, 300)，数据集规模扩大 4 倍。

ROI 截取： 将 US_N_DEPTH 从 700 缩减为 300。请在 _load_segment 中修改裁剪逻辑，只保留最核心的 300 个采样点（例如 US_DEPTH_START 之后最活跃的区间）。

1. 模型架构调整 (model.py):

UltrasoundEncoder 适配： 将 in_channels 默认值设为 1。由于输入点数变为 300，请调整 Conv1d 的 kernel_size 或 AdaptiveAvgPool1d 的参数，确保特征提取依然有效且轻量。

TCN 回归逻辑： 调整 TorqueTCN 以适配单通道 Encoder 的输出。如果预测的是相位，请在输出层考虑是否加入 Sigmoid 限制输出范围。

3. 训练逻辑同步 (train.py):

确保训练循环能正确处理 (B, W, 1, 300) 的数据输入，并适配新的相位标签。