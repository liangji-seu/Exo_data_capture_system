"""
model.py — Y型双流 TCN 髋关节力矩预测模型

结构：
  Branch A (超声空间编码器):
    每时间步 (4, 700) → Conv1d × 2 + MaxPool → Linear → 32维

  Branch B (运动学):
    每时间步 18维 直接使用

  Fusion + TCN:
    拼接 → (W, 50) → 3层膨胀因果卷积 → Linear(hidden, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 超声空间编码器（逐帧处理）────────────────────────────────────────────────

class UltrasoundEncoder(nn.Module):
    """
    输入: (B*W, 4, 700)  — batch×window 帧，每帧 4通道×700深度
    输出: (B*W, us_dim)
    """
    def __init__(self, in_channels: int = 4, us_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, padding=7),  # 包络平滑，用大核捕捉肌肉厚度
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 700 → 350
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 350 → 175
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # → 32×8 = 256
        )
        self.fc = nn.Linear(32 * 8, us_dim)

    def forward(self, x):
        # x: (B*W, 4, 700)
        h = self.conv(x)              # (B*W, 32, 8)
        h = h.flatten(1)              # (B*W, 256)
        return self.fc(h)             # (B*W, us_dim)


# ── TCN 基础块（膨胀因果卷积）────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation   # 因果填充
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _causal_trim(self, x, pad):
        """去掉因果填充在右侧多出的部分。"""
        return x[:, :, :-pad] if pad > 0 else x

    def forward(self, x):
        pad = self.conv1.padding[0]
        h = self.drop(F.relu(self.bn1(self._causal_trim(self.conv1(x), pad))))
        h = self.drop(F.relu(self.bn2(self._causal_trim(self.conv2(h), pad))))
        return F.relu(h + self.skip(x))


# ── 主模型 ────────────────────────────────────────────────────────────────────

class TorqueTCN(nn.Module):
    """
    Y型双流 TCN。

    Parameters
    ----------
    kin_dim       : 运动学特征维度（默认 18）
    us_dim        : 超声编码器输出维度（默认 32）
    tcn_hidden    : TCN 隐层通道数（默认 64）
    tcn_layers    : TCN 层数（默认 3，膨胀系数 1,2,4）
    kernel        : TCN 卷积核大小（默认 3）
    dropout       : Dropout 比例（默认 0.3）
    use_ultrasound: False 时屏蔽超声分支，仅用 IMU + 电机
    """
    def __init__(self,
                 kin_dim:       int   = 18,
                 us_dim:        int   = 32,
                 tcn_hidden:    int   = 64,
                 tcn_layers:    int   = 3,
                 kernel:        int   = 3,
                 dropout:       float = 0.3,
                 use_ultrasound: bool = True):
        super().__init__()
        self.use_ultrasound = use_ultrasound

        if use_ultrasound:
            self.us_encoder = UltrasoundEncoder(in_channels=4, us_dim=us_dim)
            fused_dim = kin_dim + us_dim   # 18 + 32 = 50
        else:
            fused_dim = kin_dim            # 18

        layers = []
        in_ch  = fused_dim
        for i in range(tcn_layers):
            dilation = 2 ** i
            layers.append(_TCNBlock(in_ch, tcn_hidden, kernel, dilation, dropout))
            in_ch = tcn_hidden
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Linear(tcn_hidden, 1)

    def forward(self,
                kin: torch.Tensor,
                us:  torch.Tensor) -> torch.Tensor:
        """
        kin : (B, W, 18)
        us  : (B, W, 4, 700)  — use_ultrasound=False 时忽略
        返回: (B, 1)
        """
        if self.use_ultrasound:
            B, W, C, D = us.shape
            us_feat = self.us_encoder(us.reshape(B * W, C, D)).reshape(B, W, -1)
            fused   = torch.cat([kin, us_feat], dim=-1)   # (B, W, 50)
        else:
            fused   = kin                                  # (B, W, 18)

        out  = self.tcn(fused.permute(0, 2, 1))           # (B, 64, W)
        return self.head(out[:, :, -1])                    # (B, 1)


# ── 快速结构验证 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, W = 4, 50
    kin = torch.randn(B, W, 18)
    us  = torch.randn(B, W, 4, 700)

    model = TorqueTCN()
    out   = model(kin, us)
    print(f"输入: kin={tuple(kin.shape)}, us={tuple(us.shape)}")
    print(f"输出: {tuple(out.shape)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
