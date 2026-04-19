"""
data_loader.py — 多模态步态数据集加载器

目录约定：
  handle_data_XX/
    input/imu.csv          — 全部 IMU 设备，按 device_id 区分
    input/motor.csv        — 电机编码器数据
    input/ultrasound.csv   — 4 通道超声，每通道 1000 采样点（原始 RF）
    label/torque.csv       — 髋关节力矩标签

超声预处理（离线，与 ultrasound_viewer.py 一致）：
  去均值 → Hilbert 包络 → log 压缩 → TGC → percentile 归一化 → 中值滤波 → 高斯平滑 → 锐化
  输出值域 [0, 1]，形状 (N, 4, 700)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace, median_filter
from scipy.signal import hilbert


# ── 常量 ──────────────────────────────────────────────────────────────────────

LEFT_IMU_SUFFIX = "260E"
US_DEPTH_START  = 150
US_DEPTH_END    = 850
US_N_CHANNELS   = 4
US_N_DEPTH      = US_DEPTH_END - US_DEPTH_START   # 700

IMU_FEATURES   = ["acc_x", "acc_y", "acc_z",
                   "gyr_x", "gyr_y", "gyr_z",
                   "mag_x", "mag_y", "mag_z",
                   "roll",  "pitch", "yaw"]        # 12 维
MOTOR_FEATURES = ["lpos", "lvel", "ltq",
                  "rpos", "rvel", "rtq"]           # 6 维
KIN_DIM = len(IMU_FEATURES) + len(MOTOR_FEATURES)  # 18 维


# ── 超声预处理（与 ultrasound_viewer.py 完全一致）────────────────────────────

def _preprocess_us_channel(raw: np.ndarray,
                            tgc_slope:   float = 0.025,
                            sigma_depth: float = 1.2,
                            sigma_time:  float = 0.8,
                            sharpen:     float = 0.2) -> np.ndarray:
    """
    raw: (N_frames, N_samples) float32  原始 RF 信号
    返回: (N_frames, N_samples) float32  包络对数压缩后归一化到 [0, 1]

    流程：
      1. 去直流（去均值）
      2. Hilbert 变换提取包络
      3. log 压缩（20·log10）
      4. TGC 深度增益补偿
      5. 逐帧 percentile 归一化到 [0, 1]
      6. 中值滤波（去椒盐噪声）
      7. 高斯平滑
      8. Laplace 锐化
    """
    raw_f = raw.astype(np.float32)
    n_frames, n_samples = raw_f.shape

    rf       = raw_f - raw_f.mean(axis=1, keepdims=True)          # 1
    envelope = np.abs(hilbert(rf, axis=1))                         # 2
    log_env  = 20.0 * np.log10(np.clip(envelope, 1e-6, None))     # 3

    out      = np.zeros_like(log_env)
    tgc_gain = np.arange(n_samples) * tgc_slope                    # 4
    for i in range(n_frames):
        frame = log_env[i] + tgc_gain
        vmin  = np.percentile(frame, 25.0)
        vmax  = np.percentile(frame, 99.0)
        out[i] = np.clip((frame - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)  # 5

    out = median_filter(out, size=(3, 3))                          # 6
    if sigma_depth > 0 or sigma_time > 0:
        out = gaussian_filter(out, sigma=[sigma_time, sigma_depth]) # 7
    if sharpen > 0:
        out = np.clip(out - sharpen * laplace(out), 0.0, 1.0)      # 8

    return out.astype(np.float32)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _nearest_align(src_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    idx      = np.searchsorted(src_ts, ref_ts)
    idx      = np.clip(idx, 0, len(src_ts) - 1)
    idx_left = np.clip(idx - 1, 0, len(src_ts) - 1)
    mask     = np.abs(src_ts[idx_left] - ref_ts) < np.abs(src_ts[idx] - ref_ts)
    idx[mask] = idx_left[mask]
    return idx


# ── 段加载 ────────────────────────────────────────────────────────────────────

def _load_segment(seg_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      kin_arr  : (N, 18)    float32  — IMU + 电机，未归一化
      us_arr   : (N, 4, 700) float32 — 超声包络，已归一化到 [0,1]
      label_arr: (N,)        float32 — 髋关节力矩 Nm/kg
    """
    # 标签（主时钟）
    torque_df = pd.read_csv(seg_dir / "label" / "torque.csv")
    ref_ts    = torque_df["timestamp"].values.astype(np.float64)
    label_arr = torque_df["hip_torque_Nm"].values.astype(np.float32)
    N = len(ref_ts)

    # IMU（260E）
    imu_df = pd.read_csv(seg_dir / "input" / "imu.csv")
    left   = imu_df[imu_df["device_id"].astype(str).str.upper()
                    .str.contains(LEFT_IMU_SUFFIX)].reset_index(drop=True)
    if len(left) == N:
        imu_vals = left[IMU_FEATURES].values.astype(np.float32)
    else:
        imu_ts   = left["timestamp"].values.astype(np.float64)
        imu_vals = left[IMU_FEATURES].values[_nearest_align(imu_ts, ref_ts)].astype(np.float32)

    # 电机
    motor_df   = pd.read_csv(seg_dir / "input" / "motor.csv")
    motor_ts   = motor_df["timestamp"].values.astype(np.float64)
    motor_vals = motor_df[MOTOR_FEATURES].values[_nearest_align(motor_ts, ref_ts)].astype(np.float32)

    kin_arr = np.concatenate([imu_vals, motor_vals], axis=1)  # (N, 18)

    # 超声：读全部 1000 列，预处理后裁剪到 [S0:S1]
    us_df  = pd.read_csv(seg_dir / "input" / "ultrasound.csv")
    all_dc = sorted([c for c in us_df.columns if c.startswith("d") and c[1:].isdigit()],
                    key=lambda x: int(x[1:]))

    us_arr = np.zeros((N, US_N_CHANNELS, US_N_DEPTH), dtype=np.float32)
    for ch_idx, ch in enumerate(range(1, US_N_CHANNELS + 1)):
        ch_df  = us_df[us_df["channel"] == ch].reset_index(drop=True)
        ch_ts  = ch_df["timestamp"].values.astype(np.float64)
        aligned = _nearest_align(ch_ts, ref_ts)
        raw    = ch_df[all_dc].values[aligned].astype(np.float32)   # (N, 1000) 原始 RF
        proc   = _preprocess_us_channel(raw)                         # (N, 1000) 包络 [0,1]
        us_arr[:, ch_idx, :] = proc[:, US_DEPTH_START:US_DEPTH_END] # 裁剪到 700 点

    return kin_arr, us_arr, label_arr


def _find_all_segments(data_root: Path) -> List[Path]:
    return sorted(data_root.rglob("handle_data_*"))


# ── 统计量 ────────────────────────────────────────────────────────────────────

def compute_kin_stats(segments: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    all_kin = np.concatenate([_load_segment(s)[0] for s in segments], axis=0)
    mean = all_kin.mean(axis=0).astype(np.float32)
    std  = all_kin.std(axis=0).astype(np.float32)
    std  = np.where(std < 1e-6, 1.0, std)
    return mean, std


def compute_label_stats(segments: List[Path]) -> Tuple[float, float]:
    all_lbl = np.concatenate([_load_segment(s)[2] for s in segments])
    return float(all_lbl.mean()), max(float(all_lbl.std()), 1e-6)


# ── Dataset ───────────────────────────────────────────────────────────────────

class GaitDataset:
    def __init__(self,
                 segments: List[Path],
                 window:   int = 50,
                 stride:   int = 1,
                 kin_mean: np.ndarray | None = None,
                 kin_std:  np.ndarray | None = None,
                 lbl_mean: float | None = None,
                 lbl_std:  float | None = None):
        self.window = window
        self.stride = stride

        self._kin_segs:     List[np.ndarray] = []
        self._us_segs:      List[np.ndarray] = []
        self._lbl_segs:     List[np.ndarray] = []
        self._sample_index: List[Tuple[int, int]] = []

        for seg_idx, seg in enumerate(segments):
            kin, us, lbl = _load_segment(seg)
            if kin_mean is not None:
                kin = (kin - kin_mean) / kin_std
            if lbl_mean is not None:
                lbl = (lbl - lbl_mean) / lbl_std
            self._kin_segs.append(kin)
            self._us_segs.append(us)
            self._lbl_segs.append(lbl)
            for start in range(0, len(lbl) - window + 1, stride):
                self._sample_index.append((seg_idx, start))

    def __len__(self):
        return len(self._sample_index)

    def __getitem__(self, idx):
        seg_idx, start = self._sample_index[idx]
        end = start + self.window
        return (self._kin_segs[seg_idx][start:end],   # (W, 18)
                self._us_segs[seg_idx][start:end],     # (W, 4, 700)
                self._lbl_segs[seg_idx][end-1:end])    # (1,)


# ── 数据集划分 ────────────────────────────────────────────────────────────────

def split_segments(all_segs: List[Path],
                   train_ratio: float = 0.67,
                   val_ratio:   float = 0.17,
                   seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    rng  = np.random.default_rng(seed)
    segs = list(all_segs)
    rng.shuffle(segs)
    n    = len(segs)
    n_tr = round(n * train_ratio)
    n_va = round(n * val_ratio)
    return segs[:n_tr], segs[n_tr:n_tr+n_va], segs[n_tr+n_va:]
