# 项目现状总结（2026-04-19）

## 任务目标

基于多模态传感器（IMU + 电机编码器 + 超声）预测髋关节力矩（Nm），用于外骨骼辅助控制。

---

## 数据现状

| 项目 | 情况 |
|------|------|
| 数据批次 | 第二批试采用（慢，中速行走） |
| 受试者 | 1人（T001） |
| 采集条件 | 零力矩，平地行走 0.8 m/s 和 1.25 m/s |
| 数据段数 | 12 段（handle_data_XX） |
| 每段帧数 | ~1100 帧（100 Hz） |
| 划分 | train=8, val=2, test=2 |
| 训练样本数 | ~10000（stride=1） |

**瓶颈：数据量严重不足，只有 1 个受试者、2 个速度条件，模型泛化能力受限。**

---

## 模型结构

Y 型双流 TCN（`model.py`）：

```
kin (B, W, 18)  ──────────────────────────────────────┐
                                                       ├→ TCN → Linear → (B,1)
us  (B, W, 4, 700) → UltrasoundEncoder(逐帧) → (B,W,32) ┘
```

- **运动学分支**：IMU 260E（12维）+ 电机编码器（6维）= 18维，Z-score 归一化
- **超声分支**：4通道 × 700深度点，经 Hilbert 包络 + log 压缩 + TGC 归一化到 [0,1]，Conv1d 编码到 32 维
- **TCN**：膨胀因果卷积，支持 `--no_ultrasound` 屏蔽超声分支
- **损失函数**：MSE + 相关系数项（`mse + 1.0*(1-corr)`），用于缓解幅值压缩问题

---

## 当前最佳结果（无超声，纯 IMU + 电机）

| 指标 | 值 |
|------|----|
| RMSE | ~6.6 Nm |
| R²   | ~0.49 |
| 训练配置 | epochs=200, batch=32, window=50, stride=1, tcn_hidden=32, tcn_layers=2, dropout=0.3, lr=1e-3 |

**加入超声后效果反而变差**，原因分析：
1. 数据量太少，超声编码器（参数量大）无法有效训练
2. 超声帧率与 IMU 不完全同步，最近邻对齐引入时间噪声
3. 需要更多条件的数据才能验证超声的贡献

---

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `train.py` | 训练，支持 `--no_ultrasound` |
| `evaluate.py` | 测试集评估，输出 RMSE/MAE/R²，`--plot` 画图 |
| `visualize.py` | 流式推理可视化（pyqtgraph），超声瀑布图 + 力矩曲线 |
| `data_loader.py` | 数据加载 + 超声预处理（Hilbert 包络） |
| `model.py` | TorqueTCN 模型定义 |

### 常用训练指令

```bash
# 无超声（当前最佳）
python train.py --data_root 超声采集数据/第二批试采用（慢，中速行走） \
  --epochs 200 --batch 32 --window 50 --stride 1 \
  --lr 1e-3 --tcn_hidden 32 --tcn_layers 2 --dropout 0.3 \
  --no_ultrasound

# 有超声
python train.py --data_root 超声采集数据/第二批试采用（慢，中速行走） \
  --epochs 200 --batch 32 --window 50 --stride 1 \
  --lr 1e-3 --tcn_hidden 32 --tcn_layers 2 --dropout 0.3

# 评估
python evaluate.py --data_root 超声采集数据/第二批试采用（慢，中速行走） \
  --ckpt checkpoints/best.pt --plot

# 流式可视化
python visualize.py \
  --seg ../超声采集数据/第二批试采用（慢，中速行走）/T001_b.零力矩_1_2.平地行走1.25m/s_20260418_131532/handle_data_01 \
  --ckpt checkpoints/best.pt --fps 30
```

---

## 下一步优先级

1. **采集更多数据**（最关键）
   - 同一受试者，增加速度条件：0.5、1.0、1.2 m/s
   - 增加重复采集次数
   - 条件允许时增加第二个受试者
   - 目标：50+ 段数据，覆盖 3+ 速度条件

2. 数据量达到后，重新验证超声分支的贡献

3. 考虑加入步态相位信息（`phase_pct` 已在 torque.csv 中）作为辅助输入
