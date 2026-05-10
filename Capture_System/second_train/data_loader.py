"""
data_loader.py — 单通道超声数据集加载器

CSV 格式（来自 gait_labeler.py）：
  d0..d299  — 原始 RF（ROI 400~700，300 点）
  torque    — 髋关节力矩 (Nm)
  phase     — 步态相位 (0~100%)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace, median_filter
from scipy.signal import hilbert

import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).parent / "data"
US_N     = 300   # ROI 内采样点数


# ── 预处理（与 gait_labeler / ultrasound_viewer 完全一致）─────────────────────

def _preprocess(raw: np.ndarray,
                tgc_slope=0.025, sigma_depth=1.2,
                sigma_time=0.8,  sharpen=0.2) -> np.ndarray:
    """raw: (N, 300) float32  →  (N, 300) float32  归一化到 [0, 1]"""
    raw_f    = raw.astype(np.float32)
    n, nsamp = raw_f.shape
    rf       = raw_f - raw_f.mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(rf, axis=1))
    log_env  = 20.0 * np.log10(np.clip(envelope, 1e-6, None))

    out      = np.zeros_like(log_env)
    tgc_gain = np.arange(nsamp) * tgc_slope
    for i in range(n):
        frame  = log_env[i] + tgc_gain
        vmin   = np.percentile(frame, 25.0)
        vmax   = np.percentile(frame, 99.0)
        out[i] = np.clip((frame - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)

    out = median_filter(out, size=(3, 3))
    if sigma_depth > 0 or sigma_time > 0:
        out = gaussian_filter(out, sigma=[sigma_time, sigma_depth])
    if sharpen > 0:
        out = np.clip(out - sharpen * laplace(out), 0.0, 1.0)
    return out.astype(np.float32)


# ── 单文件加载 ────────────────────────────────────────────────────────────────

def load_sample(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      us     : (N, 300) float32  预处理后超声包络
      torque : (N,)     float32  力矩标签
      phase  : (N,)     float32  步态相位
    """
    df     = pd.read_csv(csv_path)
    dcols  = [f"d{i}" for i in range(US_N)]
    raw    = df[dcols].values.astype(np.float32)
    torque = df["torque"].values.astype(np.float32)
    phase  = df["phase"].values.astype(np.float32)
    return _preprocess(raw), torque, phase


def find_all_samples(data_dir: Path = DATA_DIR) -> List[Path]:
    return sorted(data_dir.glob("*.csv"))


# ── Dataset ───────────────────────────────────────────────────────────────────

class UltrasoundDataset(Dataset):
    """
    滑窗数据集。每个样本：
      x   : (W, 1, 300) float32  超声窗口
      y   : (1,)         float32  窗口末帧力矩（归一化后）
    """
    def __init__(self,
                 csv_paths:    List[Path],
                 window:       int   = 50,
                 stride:       int   = 1,
                 torque_mean:  float = 0.0,
                 torque_std:   float = 1.0):
        self.window = window

        self._us_segs:     List[np.ndarray] = []
        self._torque_segs: List[np.ndarray] = []
        self._index:       List[Tuple[int, int]] = []

        for path in csv_paths:
            us, torque, _ = load_sample(path)
            torque = (torque - torque_mean) / torque_std
            self._us_segs.append(us)
            self._torque_segs.append(torque)
            n = len(torque)
            for start in range(0, n - window + 1, stride):
                self._index.append((len(self._us_segs) - 1, start))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        seg, start = self._index[idx]
        end = start + self.window
        us  = self._us_segs[seg][start:end]          # (W, 300)
        lbl = self._torque_segs[seg][end - 1:end]    # (1,)
        return (
            torch.tensor(us[:, None, :], dtype=torch.float32),  # (W, 1, 300)
            torch.tensor(lbl,            dtype=torch.float32),   # (1,)
        )


# ── 统计量 ────────────────────────────────────────────────────────────────────

def compute_torque_stats(paths: List[Path]) -> Tuple[float, float]:
    all_t = np.concatenate([load_sample(p)[1] for p in paths])
    mean  = float(all_t.mean())
    std   = max(float(all_t.std()), 1e-6)
    return mean, std


# ── 划分 ─────────────────────────────────────────────────────────────────────

def split_samples(paths:       List[Path],
                  train_ratio: float = 0.70,
                  val_ratio:   float = 0.15,
                  seed:        int   = 42
                  ) -> Tuple[List[Path], List[Path], List[Path]]:
    rng   = np.random.default_rng(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n     = len(paths)
    n_tr  = round(n * train_ratio)
    n_va  = round(n * val_ratio)
    return paths[:n_tr], paths[n_tr:n_tr + n_va], paths[n_tr + n_va:]
