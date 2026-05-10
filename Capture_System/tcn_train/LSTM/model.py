"""
model.py — 双通道中性参考 CNN-GRU 髋关节力矩预测模型

超声输入：(B, W, 2, 4, 700)
  - 2 = [当前包络, 中性参考帧]
  - 4 = 超声通道数
  - 700 = 有效深度点数

架构：
  DualCNN  : 将 2×4=8 通道送入 1D-CNN，逐帧提取空间特征 → (B*W, feat)
  GRU      : 处理帧序列的时间依赖 → 最后一步隐状态
  Head     : Linear → 力矩预测 (B, 1)

Dropout 位置：
  - CNN 每个 Conv 块后加 Dropout
  - GRU 层间 Dropout（num_layers > 1 时）
  - GRU 输出后再加一次 Dropout
"""

import torch
import torch.nn as nn


class DualCNN(nn.Module):
    """
    逐帧 1D-CNN，输入 2×4=8 通道（当前帧 + 参考帧拼接）。
    (B, 8, 700) → (B, feat_dim)
    """

    def __init__(self, feat_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层：大感受野，捕捉肌肉回波的宽峰结构
            nn.Conv1d(8, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),                        # 700 → 350

            # 第二层：中等感受野
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),                        # 350 → 175

            # 第三层：细粒度特征
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),                # → (B, 128, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, 700) → (B, feat_dim)"""
        return self.proj(self.net(x).squeeze(-1))


class TorqueCNNGRU(nn.Module):
    """
    参数
    ----
    kin_dim      : 运动学特征维度（IMU 12 + 电机 6 = 18）
    us_feat      : CNN 输出维度
    gru_hidden   : GRU 隐层维度
    gru_layers   : GRU 层数
    dropout      : Dropout 概率
    use_ultrasound : False 时屏蔽超声分支，仅用运动学
    """

    def __init__(self,
                 kin_dim:       int   = 18,
                 us_feat:       int   = 64,
                 gru_hidden:    int   = 128,
                 gru_layers:    int   = 2,
                 dropout:       float = 0.3,
                 use_ultrasound: bool = True):
        super().__init__()
        self.use_ultrasound = use_ultrasound

        if use_ultrasound:
            # 8 = 2（当前+参考）× 4（超声通道）
            self.cnn   = DualCNN(feat_dim=us_feat, dropout=dropout * 0.7)
            gru_in     = kin_dim + us_feat
        else:
            gru_in     = kin_dim

        self.gru = nn.GRU(
            input_size  = gru_in,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, 1)

    def forward(self, kin: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        """
        kin : (B, W, 18)
        us  : (B, W, 2, 4, 700)
        返回: (B, 1)
        """
        if self.use_ultrasound:
            B, W, two, C, D = us.shape
            # 把 2×4 通道合并 → (B*W, 8, 700)
            us_flat = us.reshape(B * W, two * C, D)
            us_feat = self.cnn(us_flat).reshape(B, W, -1)   # (B, W, us_feat)
            fused   = torch.cat([kin, us_feat], dim=-1)      # (B, W, kin+us)
        else:
            fused = kin

        out, _ = self.gru(fused)                             # (B, W, gru_hidden)
        return self.head(self.drop(out[:, -1, :]))           # (B, 1)
