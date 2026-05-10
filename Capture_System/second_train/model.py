"""
model.py — 单通道超声 TCN 力矩预测模型

输入 : (B, W, 1, 300)  超声包络（单通道，300 深度点）
输出 : (B, 1)           归一化力矩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 超声空间编码器（逐帧）────────────────────────────────────────────────────

class UltrasoundEncoder(nn.Module):
    """
    输入: (B*W, 1, 300)
    输出: (B*W, us_dim)

    用大核捕捉肌肉厚度包络，AdaptiveAvgPool 保证输出定长。
    """
    def __init__(self, us_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),   # 300
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # 150
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # 75
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),                        # (32, 8) = 256
        )
        self.fc = nn.Linear(32 * 8, us_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


# ── TCN 基础块（膨胀因果卷积）────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.conv1.padding[0]
        h = self.drop(F.relu(self.bn1(self._trim(self.conv1(x), pad))))
        h = self.drop(F.relu(self.bn2(self._trim(self.conv2(h), pad))))
        return F.relu(h + self.skip(x))

    @staticmethod
    def _trim(x: torch.Tensor, pad: int) -> torch.Tensor:
        return x[:, :, :-pad] if pad > 0 else x


# ── 主模型 ────────────────────────────────────────────────────────────────────

class UltraTCN(nn.Module):
    """
    单通道超声 → TCN → 力矩预测。

    Parameters
    ----------
    us_dim     : 超声编码器输出维度（默认 32）
    tcn_hidden : TCN 隐层通道数（默认 64）
    tcn_layers : TCN 层数，膨胀系数 1,2,4,8,...（默认 4）
    kernel     : 因果卷积核大小（默认 3）
    dropout    : Dropout 比例（默认 0.2）
    """
    def __init__(self,
                 us_dim:     int   = 32,
                 tcn_hidden: int   = 64,
                 tcn_layers: int   = 4,
                 kernel:     int   = 3,
                 dropout:    float = 0.2):
        super().__init__()
        self.encoder = UltrasoundEncoder(us_dim=us_dim)

        layers, in_ch = [], us_dim
        for i in range(tcn_layers):
            layers.append(_TCNBlock(in_ch, tcn_hidden, kernel, 2 ** i, dropout))
            in_ch = tcn_hidden
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Linear(tcn_hidden, 1)

    def forward(self, us: torch.Tensor) -> torch.Tensor:
        """
        us  : (B, W, 1, 300)
        返回: (B, 1)
        """
        B, W, C, D = us.shape
        feat = self.encoder(us.reshape(B * W, C, D)).reshape(B, W, -1)  # (B, W, us_dim)
        out  = self.tcn(feat.permute(0, 2, 1))                           # (B, hidden, W)
        return self.head(out[:, :, -1])                                   # (B, 1)


# ── 快速验证 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, W = 4, 50
    us   = torch.randn(B, W, 1, 300)
    m    = UltraTCN()
    out  = m(us)
    print(f"输入 : {tuple(us.shape)}")
    print(f"输出 : {tuple(out.shape)}")
    print(f"参数量: {sum(p.numel() for p in m.parameters()):,}")
