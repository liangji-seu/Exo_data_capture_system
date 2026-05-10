"""
dataset.py — 双通道中性参考数据集

超声输入格式：将"当前包络帧"和"中性参考帧"拼接为 2×4×700 张量。
  - 通道 0：当前包络（归一化到 [0,1]）
  - 通道 1：中性参考帧（同一受试者静息态均值，广播到每帧）

这样 CNN 可以直接学习"当前 vs 参考"的差异，而不是只看差分值，
保留了绝对幅值信息，同时让模型自己决定如何利用参考。

数据增强（训练时）：
  - 随机深度遮蔽（Blank Portions）
  - 通道随机偏移（Channel Shifting）
  - 高斯噪声
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))        # LSTM/ 自身
sys.path.insert(1, str(Path(__file__).parent.parent))  # tcn_train/ for data_loader
from data_loader import (
    _load_segment, _nearest_align,
    _preprocess_us_channel,
    compute_kin_stats, compute_label_stats, KIN_DIM,
    _find_all_segments, split_segments,
)

US_S0, US_S1 = 150, 850   # 有效深度 → 700 点
US_D = US_S1 - US_S0      # 700


# ── 超声加载：返回包络 + 参考帧 ───────────────────────────────────────────────

def _load_us_dual(seg_dir: Path,
                  ref_ts: np.ndarray,
                  ref_frames: int = 30
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      us_cur : (N, 4, 700)  当前包络，[0,1]
      us_ref : (4, 700)     中性参考帧（前 ref_frames 帧均值）
    """
    us_df = pd.read_csv(seg_dir / "input" / "ultrasound.csv")
    dcols = sorted([c for c in us_df.columns if c.startswith("d") and c[1:].isdigit()],
                   key=lambda x: int(x[1:]))
    N = len(ref_ts)

    us_cur = np.zeros((N, 4, US_D), dtype=np.float32)
    us_ref = np.zeros((4, US_D),    dtype=np.float32)

    for ch_idx, ch in enumerate(range(1, 5)):
        sub   = us_df[us_df["channel"] == ch].reset_index(drop=True)
        ch_ts = sub["timestamp"].values.astype(np.float64)
        idx   = _nearest_align(ch_ts, ref_ts)
        raw   = sub[dcols].values[idx].astype(np.float32)          # (N, 1000)
        env   = _preprocess_us_channel(raw)[:, US_S0:US_S1]        # (N, 700)
        us_cur[:, ch_idx, :] = env
        us_ref[ch_idx]       = env[:ref_frames].mean(axis=0)       # (700,)

    return us_cur, us_ref


# ── 数据增强 ──────────────────────────────────────────────────────────────────

def _augment(us: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    us: (W, 4, 700) 当前包络帧序列（增强只作用于当前帧，不动参考帧）
    """
    us = us.copy()
    D = us.shape[-1]

    if rng.random() < 0.5:                          # 随机深度遮蔽
        mask_len = rng.integers(int(D * 0.05), int(D * 0.15))
        start    = rng.integers(0, D - mask_len)
        us[:, :, start:start + mask_len] = 0.0

    if rng.random() < 0.4:                          # 通道随机偏移
        shift = rng.integers(-10, 11)
        us = np.roll(us, shift, axis=-1)

    if rng.random() < 0.5:                          # 高斯噪声
        us = np.clip(us + rng.normal(0, 0.02, us.shape).astype(np.float32), 0.0, 1.0)

    return us


# ── Dataset ───────────────────────────────────────────────────────────────────

class DualChannelDataset(Dataset):
    """
    滑动窗口数据集。

    每个样本的超声输入形状：(W, 2, 4, 700)
      dim-1 = 0 : 当前包络帧
      dim-1 = 1 : 中性参考帧（广播，每帧相同）

    模型可以把 2×4 = 8 个通道送入 CNN，或者分别处理再融合。
    """

    def __init__(self,
                 segments:   List[Path],
                 window:     int   = 50,
                 stride:     int   = 1,
                 kin_mean:   np.ndarray | None = None,
                 kin_std:    np.ndarray | None = None,
                 lbl_mean:   float | None = None,
                 lbl_std:    float | None = None,
                 ref_frames: int   = 30,
                 augment:    bool  = False,
                 seed:       int   = 42):
        self.window  = window
        self.augment = augment
        self.rng     = np.random.default_rng(seed)

        self._kin_segs: List[np.ndarray] = []
        self._us_segs:  List[np.ndarray] = []   # (N, 2, 4, 700)
        self._lbl_segs: List[np.ndarray] = []
        self._index:    List[Tuple[int, int]] = []

        for seg_idx, seg in enumerate(segments):
            kin, _, lbl = _load_segment(seg)

            torque_df = pd.read_csv(seg / "label" / "torque.csv")
            ref_ts    = torque_df["timestamp"].values.astype(np.float64)

            us_cur, us_ref = _load_us_dual(seg, ref_ts, ref_frames)
            # 参考帧广播到每一帧：(N, 4, 700)
            us_ref_broad = np.broadcast_to(us_ref[np.newaxis], us_cur.shape).copy()
            # 拼成双通道：(N, 2, 4, 700)
            us_dual = np.stack([us_cur, us_ref_broad], axis=1)

            if kin_mean is not None:
                kin = (kin - kin_mean) / kin_std
            if lbl_mean is not None:
                lbl = (lbl - lbl_mean) / lbl_std

            self._kin_segs.append(kin)
            self._us_segs.append(us_dual)
            self._lbl_segs.append(lbl)

            for start in range(0, len(lbl) - window + 1, stride):
                self._index.append((seg_idx, start))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        seg_idx, start = self._index[idx]
        end = start + self.window

        kin    = self._kin_segs[seg_idx][start:end].copy()    # (W, 18)
        us     = self._us_segs[seg_idx][start:end].copy()     # (W, 2, 4, 700)
        lbl    = self._lbl_segs[seg_idx][end - 1:end].copy()  # (1,)

        if self.augment:
            # 只增强当前帧（通道 0），参考帧保持不变
            us[:, 0, :, :] = _augment(us[:, 0, :, :], self.rng)

        return (torch.from_numpy(kin),
                torch.from_numpy(us),
                torch.from_numpy(lbl))
