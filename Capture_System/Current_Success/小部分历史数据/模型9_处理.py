import os
import re
import shutil
import importlib.util
import glob
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE_TRAIN_SCRIPT = r"D:\nightmare\Documents\SRTP\LSTM\模型9_增强.py"

CONFIG = {
    "data_dir": r"D:\nightmare\Documents\SRTP\AllData\Processed_Final_Training_Set_clean_v2_fix_shift_v2",
    "save_dir": r"D:\nightmare\Documents\SRTP\LSTM\training_logs\ultr_fusion_model_9_auto_v6_1",
    "quality_report_path": r"D:\nightmare\Documents\SRTP\AllData\Merge_Data_Process\pipeline_run_20260313_204128\03_reports\clean_set_build\clean_build_report.csv",
    "k_folds": 5,
    "cv_strategy": "balanced_group",  # balanced_group | groupkfold
    "cv_shuffle_groups": True,
    "cv_random_seed": 42,
    # [修改点 1] 增加序列长度，扩大模型感受野，让它提前“看到”跳变先兆
    "sequence_length": 48,
    "pca_components": 20,
    # 标准化策略:
    # - global: 全量训练集统一 StandardScaler（旧逻辑）
    # - trial : 每个 trial 自身做 z-score（更适合跨天/跨域数据）
    "normalization_mode": "trial",
    "augment_factors": [0.8, 0.9, 1.0, 1.1],
    "epochs": 150,
    "batch_size": 32,
    "learning_rate": 0.0015,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 6,
    "early_stop_patience": 25,
    "smoothing_strategy": "adaptive_tracker",  # 回退到 v5_7 表现更稳的默认策略
    "strategy_eval_enable": True,
    "strategy_candidates": [
        "adaptive_tracker",
        "robust_tracker_v2",
        "smart_tracker",
        "raw_monotonic",
        "robust_tracker",
        "lowpass",
        "raw",
        "hybrid_ensemble",
    ],
    # 综合评分 = 主 MAE + edge*w1 + jump*w2 + spike*w3 + slope*w4（越低越好）
    "strategy_score_edge_weight": 0.15,
    "strategy_score_jump_weight": 0.05,
    "strategy_score_spike_weight": 0.05,
    "strategy_score_slope_weight": 0.08,
    "edge_weight_enable": True,
    # 此处的 edge_ratio 代表相位在[0, 0.15] 和 [0.85, 1.0] 范围内时启用加权
    "edge_ratio": 0.15,
    "edge_dynamic_enable": False,
    "edge_target_frames": 10,
    "edge_ratio_min": 0.08,
    "edge_ratio_max": 0.24,
    "edge_weight": 3.5,  # 回调到更稳的边界权重
    # [修改点 2] 大幅提高 tracker_alpha，减少历史动量带来的相位滞后（从 0.25 -> 0.85）
    "tracker_alpha": 0.75,
    "tracker_beta": 0.03,
    # robust_tracker_v2 限制单步相位变化，减少尖跳
    "tracker_max_step": 0.12,
    "adaptive_vel_low": 0.035,
    "adaptive_vel_high": 0.120,
    "adaptive_blend_min": 0.10,
    "adaptive_blend_max": 0.85,
    "adaptive_conf_center": 0.85,
    "adaptive_conf_width": 0.30,
    "wrap_hysteresis_enable": False,
    "wrap_arm_high": 0.90,
    "wrap_release_low": 0.10,
    "anti_spike_enable": False,
    "anti_spike_mad_k": 3.0,
    "anti_spike_abs_step": 0.06,
    "lag_comp_enable": True,
    "lag_comp_max_shift": 6,
    "plot_use_lag_comp_phase": False,
    # Strict evaluation:
    # - no_lag: main reported MAE does not use per-trial oracle lag search
    # - oracle_lag: keep legacy behavior as primary metric
    "strict_eval_enable": True,
    "strict_eval_primary": "no_lag",  # no_lag | oracle_lag
    "strict_disable_external_lag_penalty": True,
    "diag_jump_threshold": 0.08,
    "diag_spike_threshold": 0.06,
    "mono_min_step": -0.005,
    "mono_max_step": 0.080,
    "ema_alpha": 0.6,
    "butter_order": 3,
    "butter_cutoff": 0.1,
    # quality scoring
    "sensor_low_clip": 55.0,
    "sensor_high_clip": 205.0,
    "sat_warn_threshold": 0.08,
    "sat_bad_threshold": 0.22,
    "phase_coverage_min": 0.82,
    "effective_cycles_min": 4.0,
    "quality_floor": 0.70,
    "quality_ceiling": 1.10,
    "domain_balance_enable": False,
    "manual_weight_overrides_enable": True,
    # Domain-specific conservative downweight (for harder/noisier domains).
    "domain_weight_overrides": {
        "Processed_Final_Training_Set_1.14_1": 0.85,
    },
    "domain_weight_min": 0.75,
    "domain_weight_max": 1.20,
    # Short-clip controls:
    # - penalty: always enabled in quality scoring
    # - hard drop: optional switch for strict filtering
    "short_clip_warn_frames": 120,
    "short_clip_drop_frames": 95,
    "short_clip_penalty_weight": 0.55,
    "drop_short_trials_enable": True,
    "drop_short_trials_frames": 110,
    # Subject-level hard-example reweighting:
    # keys are subject IDs parsed from "Sub_xx" in filenames.
    # B方案：对反复进入最差样本榜的受试者做保守降权
    "subject_weight_map": {
        "11": 0.85,
        "15": 0.80,
        "18": 0.92,
        "23": 0.90,
    },
    "subject_weight_default": 1.00,
    "subject_weight_min": 0.80,
    "subject_weight_max": 1.50,
    "sample_weight_min": 0.70,
    "sample_weight_max": 1.30,
    "lag_penalty_enable": False,
    "lag_risk_report_path": "",
    "lag_penalty_abs_shift": 4,
    "lag_penalty_weight": 0.85,
    "lag_penalty_max_ratio": 0.35,  # 防止单折大面积降权导致欠拟合
    "auto_calibrate_postprocess": False,
    "auto_calibrate_verbose": True,
    "auto_calibrate_mode": "guarded",  # guarded | legacy
    "auto_calibrate_min_trials": 24,
    "auto_calibrate_cycle_cv_max": 0.22,
    "auto_calibrate_max_rel_change": 0.20,
    "random_seed": 42,
}


class AdaptiveScaler:
    """
    Keep a scikit-like scaler interface while supporting per-trial normalization.
    """
    def __init__(self, mode="global", eps=1e-6):
        self.mode = mode
        self.eps = eps
        self._global_scaler = StandardScaler() if mode == "global" else None

    def fit(self, x):
        if self.mode == "global":
            self._global_scaler.fit(x)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "global":
            return self._global_scaler.transform(x)

        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        return (x - mean) / (std + self.eps)

    def fit_transform(self, x):
        return self.fit(x).transform(x)


def apply_env_overrides():
    data_dir = os.environ.get("M9_DATA_DIR")
    quality_report_path = os.environ.get("M9_QUALITY_REPORT_PATH")
    save_dir = os.environ.get("M9_SAVE_DIR")
    if data_dir:
        CONFIG["data_dir"] = data_dir
        print(f"[ENV] 覆盖 data_dir -> {data_dir}")
    if quality_report_path:
        CONFIG["quality_report_path"] = quality_report_path
        print(f"[ENV] 覆盖 quality_report_path -> {quality_report_path}")
    if save_dir:
        CONFIG["save_dir"] = save_dir
        print(f"[ENV] 覆盖 save_dir -> {save_dir}")


def load_base_module(path: str):
    spec = importlib.util.spec_from_file_location("base_train_module_weighted_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础训练脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_subject_id(name: str) -> str:
    m = re.search(r"Sub_(\d+)", name, flags=re.IGNORECASE)
    return m.group(1) if m else f"UNK_{name}"


def parse_domain(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[0]
    if "1.14" in name:
        return "set_1.14"
    return "set_1"


def phase_from_labels(labels: np.ndarray) -> np.ndarray:
    phase = np.arctan2(labels[:, 0], labels[:, 1])
    return (phase + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)


def trial_quality_score(features: np.ndarray, labels: np.ndarray):
    phase = phase_from_labels(labels)
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    n_frames = int(len(labels))
    effective_cycles = float(unwrapped[-1] - unwrapped[0]) if len(unwrapped) > 1 else 0.0
    phase_coverage = float(np.quantile(phase, 0.95) - np.quantile(phase, 0.05))
    sat_ratio = float(((features <= CONFIG["sensor_low_clip"]) | (features >= CONFIG["sensor_high_clip"])).mean())

    score = 1.0
    if sat_ratio > CONFIG["sat_warn_threshold"]:
        if sat_ratio >= CONFIG["sat_bad_threshold"]:
            score *= 0.70
        else:
            score *= max(0.80, 1.0 - (sat_ratio - CONFIG["sat_warn_threshold"]) * 4.0)
    if phase_coverage < CONFIG["phase_coverage_min"]:
        score *= max(0.60, phase_coverage / CONFIG["phase_coverage_min"])
    if effective_cycles < CONFIG["effective_cycles_min"]:
        score *= max(0.60, effective_cycles / CONFIG["effective_cycles_min"])

    # Penalize very short clips because they are empirically unstable in validation.
    warn_f = int(CONFIG.get("short_clip_warn_frames", 120))
    drop_f = int(CONFIG.get("short_clip_drop_frames", 95))
    low_w = float(CONFIG.get("short_clip_penalty_weight", 0.55))
    if n_frames < warn_f:
        span = max(1, warn_f - drop_f)
        alpha = np.clip((n_frames - drop_f) / span, 0.0, 1.0)
        len_factor = low_w + (1.0 - low_w) * alpha
        score *= len_factor

    score = float(np.clip(score, CONFIG["quality_floor"], CONFIG["quality_ceiling"]))
    return score, sat_ratio, phase_coverage, effective_cycles


def _trial_key(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def phase_from_preds(preds_raw: np.ndarray) -> np.ndarray:
    raw_rad = np.arctan2(preds_raw[:, 0], preds_raw[:, 1])
    return (raw_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)


def circular_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    d = np.where(d > 0.5, d - 1.0, d)
    d = np.where(d < -0.5, d + 1.0, d)
    return d


def circular_mae(pred: np.ndarray, true: np.ndarray) -> float:
    diff = np.abs(circular_delta(pred, true))
    return float(np.mean(diff)) if len(diff) else float("nan")


def shift_phase_with_edge_hold(phase: np.ndarray, shift: int) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    if shift == 0 or len(phase) == 0:
        return phase.copy()
    out = np.empty_like(phase)
    if shift > 0:
        out[:-shift] = phase[shift:]
        out[-shift:] = phase[-1]
    else:
        k = -shift
        out[k:] = phase[:-k]
        out[:k] = phase[0]
    return out


def best_lag_compensation(pred_phase: np.ndarray, true_phase: np.ndarray, max_shift: int):
    pred_phase = np.asarray(pred_phase, dtype=np.float32)
    true_phase = np.asarray(true_phase, dtype=np.float32)
    if len(pred_phase) == 0 or len(true_phase) == 0:
        return pred_phase, 0, float("nan")

    best_shift = 0
    best_mae = circular_mae(pred_phase, true_phase)
    best_phase = pred_phase.copy()
    for shift in range(-max_shift, max_shift + 1):
        shifted = shift_phase_with_edge_hold(pred_phase, shift)
        mae = circular_mae(shifted, true_phase)
        if mae < best_mae:
            best_mae = mae
            best_shift = shift
            best_phase = shifted
    return best_phase, int(best_shift), float(best_mae)


def edge_phase_mae(pred_phase: np.ndarray, true_phase: np.ndarray, edge_ratio: float) -> float:
    p = np.asarray(pred_phase, dtype=np.float32)
    t = np.asarray(true_phase, dtype=np.float32)
    edge_mask = (t <= edge_ratio) | (t >= 1.0 - edge_ratio)
    if not np.any(edge_mask):
        return float("nan")
    return circular_mae(p[edge_mask], t[edge_mask])


def estimate_cycle_frames(phase: np.ndarray) -> float:
    phase = np.asarray(phase, dtype=np.float32)
    if len(phase) < 3:
        return float("nan")
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    cycles = float(unwrapped[-1] - unwrapped[0])
    if cycles <= 1e-6:
        return float("nan")
    return float(len(phase) / cycles)


def dynamic_edge_ratio_from_phase(phase: np.ndarray) -> float:
    base_ratio = float(CONFIG.get("edge_ratio", 0.15))
    if not bool(CONFIG.get("edge_dynamic_enable", True)):
        return base_ratio
    cycle_frames = estimate_cycle_frames(phase)
    if not np.isfinite(cycle_frames):
        return base_ratio
    target_frames = float(CONFIG.get("edge_target_frames", 10))
    ratio = target_frames / max(1.0, cycle_frames)
    return float(
        np.clip(
            ratio,
            CONFIG.get("edge_ratio_min", 0.08),
            CONFIG.get("edge_ratio_max", 0.24),
        )
    )


def slope_phase_mae(pred_phase: np.ndarray, true_phase: np.ndarray) -> float:
    p = np.asarray(pred_phase, dtype=np.float32)
    t = np.asarray(true_phase, dtype=np.float32)
    if len(p) < 2 or len(t) < 2:
        return float("nan")
    dp = circular_delta(p[1:], p[:-1])
    dt = circular_delta(t[1:], t[:-1])
    return float(np.mean(np.abs(dp - dt)))


def jump_rate(phase: np.ndarray, threshold: float) -> float:
    phase = np.asarray(phase, dtype=np.float32)
    if len(phase) < 2:
        return 0.0
    step = np.abs(circular_delta(phase[1:], phase[:-1]))
    return float(np.mean(step > threshold))


def spike_rate(phase: np.ndarray, threshold: float) -> float:
    phase = np.asarray(phase, dtype=np.float32)
    if len(phase) < 2:
        return 0.0
    step = np.abs(circular_delta(phase[1:], phase[:-1]))
    return float(np.mean(step > threshold))


def suppress_phase_spikes(phase: np.ndarray, mad_k: float, abs_step_th: float) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    if len(phase) < 4:
        return phase.copy()

    steps = circular_delta(phase[1:], phase[:-1]).astype(np.float32)
    pad = np.pad(steps, (1, 1), mode="edge")
    med = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]], axis=1), axis=1).astype(np.float32)
    residual = steps - med
    mad = float(np.median(np.abs(residual)))
    limit = max(1e-4, mad_k * mad)
    is_spike = (np.abs(residual) > limit) & (np.abs(steps) > abs_step_th)
    fixed_steps = np.where(is_spike, med + np.clip(residual, -limit, limit), steps)

    out = np.empty_like(phase)
    out[0] = phase[0]
    for i in range(len(fixed_steps)):
        out[i + 1] = (out[i] + fixed_steps[i]) % 1.0
    return out


def circular_mean_stack(phases, weights=None) -> np.ndarray:
    mats = [np.atleast_1d(np.asarray(p, dtype=np.float32)) for p in phases if p is not None]
    if not mats:
        return np.array([], dtype=np.float32)
    if len(mats) == 1:
        return mats[0].astype(np.float32)

    arr = np.stack(mats, axis=0)  # [K, T]
    if arr.ndim == 1:
        arr = arr[:, None]
    if weights is None:
        w2d = np.ones((arr.shape[0], arr.shape[1]), dtype=np.float32) / float(arr.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(w) != arr.shape[0]:
            raise ValueError("weights length must match number of phase tracks.")
        s = float(np.sum(w))
        w = np.ones_like(w) / float(len(w)) if s <= 1e-8 else (w / s)
        w2d = np.repeat(w[:, None], arr.shape[1], axis=1)

    sinv = np.sin(arr * 2 * np.pi)
    cosv = np.cos(arr * 2 * np.pi)
    mean_sin = np.sum(sinv * w2d, axis=0)
    mean_cos = np.sum(cosv * w2d, axis=0)
    return ((np.arctan2(mean_sin, mean_cos) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)).astype(np.float32)


def smooth_raw_monotonic(preds_raw: np.ndarray, min_step: float, max_step: float) -> np.ndarray:
    phase = phase_from_preds(preds_raw)
    if len(phase) < 3:
        return phase.astype(np.float32)

    steps = circular_delta(phase[1:], phase[:-1]).astype(np.float32)
    pad = np.pad(steps, (1, 1), mode="edge")
    med = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]], axis=1), axis=1).astype(np.float32)
    # Keep gait phase mostly monotonic while tolerating tiny local fluctuations.
    clean_steps = np.clip(med, min_step, max_step)

    out = np.empty_like(phase, dtype=np.float32)
    out[0] = float(phase[0])
    for i in range(len(clean_steps)):
        out[i + 1] = (out[i] + clean_steps[i]) % 1.0
    return out


def auto_calibrate_postprocess(scored_trials, tag: str = "global"):
    if not bool(CONFIG.get("auto_calibrate_postprocess", True)):
        return
    step_abs_list = []
    cycle_frames = []
    for _, _, label, *_rest in scored_trials:
        cyc = _rest[-1]  # effective_cycles
        phase = phase_from_labels(label)
        if len(phase) > 1:
            d = np.abs(circular_delta(phase[1:], phase[:-1]))
            step_abs_list.append(d.astype(np.float32))
        if cyc and cyc > 1e-6:
            cycle_frames.append(float(len(label)) / float(cyc))
    if not step_abs_list:
        return

    step_abs = np.concatenate(step_abs_list)
    q50, q90, q95 = np.quantile(step_abs, [0.50, 0.90, 0.95])
    raw_mono_max = float(np.clip(q95 * 1.35, 0.05, 0.14))
    raw_mono_min_abs = float(np.clip(q50 * 0.35, 0.002, 0.02))
    raw_jump_th = float(np.clip(q90 * 1.15, 0.06, 0.12))
    raw_spike_th = float(np.clip(q95 * 0.90, 0.05, 0.11))
    raw_lag_max = int(np.clip(round((np.median(cycle_frames) * 0.22) if cycle_frames else 6), 3, 10))

    mode = str(CONFIG.get("auto_calibrate_mode", "guarded")).lower()
    if mode != "legacy":
        n_trials = len(scored_trials)
        min_trials = int(CONFIG.get("auto_calibrate_min_trials", 24))
        if n_trials < min_trials:
            if bool(CONFIG.get("auto_calibrate_verbose", True)):
                print(f"[AutoCalib] 跳过: trial 数不足 ({n_trials} < {min_trials})，保持手动基线参数。")
            return
        if len(cycle_frames) >= 4:
            cyc_cv = float(np.std(cycle_frames) / max(1e-6, np.mean(cycle_frames)))
            if cyc_cv > float(CONFIG.get("auto_calibrate_cycle_cv_max", 0.22)):
                if bool(CONFIG.get("auto_calibrate_verbose", True)):
                    print(
                        f"[AutoCalib] 跳过: 周期分布离散过大 (cv={cyc_cv:.3f})，"
                        "保持手动基线参数。"
                    )
                return

    def _bounded_rel(anchor, proposed, lo, hi):
        max_rel = float(CONFIG.get("auto_calibrate_max_rel_change", 0.20))
        low = anchor * (1.0 - max_rel)
        high = anchor * (1.0 + max_rel)
        if low > high:
            low, high = high, low
        return float(np.clip(proposed, max(lo, low), min(hi, high)))

    # Anchor around manual-tuned baseline to prevent aggressive drift.
    anchor_mono_max = float(CONFIG.get("mono_max_step", 0.08))
    anchor_mono_min_abs = abs(float(CONFIG.get("mono_min_step", -0.005)))
    anchor_jump = float(CONFIG.get("diag_jump_threshold", 0.08))
    anchor_spike = float(CONFIG.get("diag_spike_threshold", 0.06))
    anchor_lag = int(CONFIG.get("lag_comp_max_shift", 6))

    if mode == "legacy":
        mono_max = raw_mono_max
        mono_min = -raw_mono_min_abs
        jump_th = raw_jump_th
        spike_th = raw_spike_th
        lag_max = raw_lag_max
    else:
        mono_max = _bounded_rel(anchor_mono_max, raw_mono_max, 0.05, 0.14)
        mono_min_abs = _bounded_rel(anchor_mono_min_abs, raw_mono_min_abs, 0.002, 0.02)
        mono_min = -mono_min_abs
        jump_th = _bounded_rel(anchor_jump, raw_jump_th, 0.06, 0.12)
        spike_th = _bounded_rel(anchor_spike, raw_spike_th, 0.05, 0.11)
        # Keep spike threshold no looser than jump threshold.
        spike_th = min(spike_th, jump_th)
        lag_max = int(np.clip(round(_bounded_rel(float(anchor_lag), float(raw_lag_max), 4, 8)), 4, 8))

    CONFIG["mono_max_step"] = mono_max
    CONFIG["mono_min_step"] = mono_min
    CONFIG["diag_jump_threshold"] = jump_th
    CONFIG["diag_spike_threshold"] = spike_th
    CONFIG["lag_comp_max_shift"] = lag_max

    if bool(CONFIG.get("auto_calibrate_verbose", True)):
        print(
            "[AutoCalib] mode={} | mono_max_step={:.4f}, mono_min_step={:.4f}, "
            "jump_th={:.4f}, spike_th={:.4f}, lag_comp_max_shift={}".format(
                mode, mono_max, mono_min, jump_th, spike_th, lag_max
            )
        )
    try:
        snapshot = {
            "mode": mode,
            "n_trials": int(len(scored_trials)),
            "raw": {
                "mono_max_step": raw_mono_max,
                "mono_min_abs": raw_mono_min_abs,
                "jump_threshold": raw_jump_th,
                "spike_threshold": raw_spike_th,
                "lag_comp_max_shift": raw_lag_max,
            },
            "applied": {
                "mono_max_step": float(mono_max),
                "mono_min_step": float(mono_min),
                "jump_threshold": float(jump_th),
                "spike_threshold": float(spike_th),
                "lag_comp_max_shift": int(lag_max),
            },
        }
        suffix = f"_{tag}" if tag else ""
        out_path = os.path.join(CONFIG["save_dir"], f"auto_calibration_snapshot{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_balanced_group_splits(groups, n_splits, random_seed=42, shuffle_groups=True):
    groups = list(groups)
    idx_all = np.arange(len(groups))
    group_counts = Counter(groups)
    unique_groups = list(group_counts.keys())
    if n_splits < 2 or len(unique_groups) < n_splits:
        raise ValueError(f"分折失败：groups={len(unique_groups)}, n_splits={n_splits}")

    rng = np.random.default_rng(random_seed)
    ranked_groups = sorted(unique_groups, key=lambda g: (-group_counts[g], str(g)))
    if shuffle_groups:
        # Shuffle first, then stable-sort by size to keep balancing while removing deterministic fold IDs.
        rng.shuffle(ranked_groups)
        ranked_groups = sorted(ranked_groups, key=lambda g: (-group_counts[g], str(g)))

    fold_groups = [set() for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    for g in ranked_groups:
        min_sz = min(fold_sizes)
        candidates = [i for i, sz in enumerate(fold_sizes) if sz == min_sz]
        fidx = int(rng.choice(candidates)) if len(candidates) > 1 else candidates[0]
        fold_groups[fidx].add(g)
        fold_sizes[fidx] += int(group_counts[g])

    splits = []
    for fidx in range(n_splits):
        val_mask = np.array([g in fold_groups[fidx] for g in groups], dtype=bool)
        val_idx = idx_all[val_mask]
        train_idx = idx_all[~val_mask]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, val_idx))

    return splits, fold_groups, fold_sizes


def _smooth_robust_tracker_v2(base, preds_raw: np.ndarray) -> np.ndarray:
    tracker = base.RobustPhaseTracker(alpha=CONFIG["tracker_alpha"], beta=CONFIG["tracker_beta"])
    phases = []
    max_step = float(CONFIG.get("tracker_max_step", 0.12))
    wrap_hyst = bool(CONFIG.get("wrap_hysteresis_enable", True))
    wrap_arm_high = float(CONFIG.get("wrap_arm_high", 0.90))
    wrap_release_low = float(CONFIG.get("wrap_release_low", 0.10))
    wrap_armed = False
    for s, c in preds_raw:
        p = tracker.update(s, c)
        if not phases:
            phases.append(p)
            continue

        prev = phases[-1]
        delta = p - prev
        if delta > 0.5:
            delta -= 1.0
        elif delta < -0.5:
            delta += 1.0

        if wrap_hyst:
            if prev >= wrap_arm_high:
                wrap_armed = True
            wrap_reset = wrap_armed and (p <= wrap_release_low)
            if wrap_reset:
                wrap_armed = False
            elif prev < 0.5 and p > 0.5:
                # Reject accidental reverse crossing when already near low phase.
                wrap_armed = False
        else:
            wrap_reset = prev > 0.85 and p < 0.15
        if not wrap_reset:
            delta = float(np.clip(delta, -max_step, max_step))
            p = (prev + delta) % 1.0
        phases.append(p)
    out = np.asarray(phases, dtype=np.float32)
    if bool(CONFIG.get("anti_spike_enable", True)):
        out = suppress_phase_spikes(
            out,
            mad_k=float(CONFIG.get("anti_spike_mad_k", 3.0)),
            abs_step_th=float(CONFIG.get("anti_spike_abs_step", 0.06)),
        )
    return out


def load_external_quality_map(path: str):
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] 无法读取 quality_report_path: {e}")
        return {}

    required = {"file", "quality_score"}
    if not required.issubset(set(df.columns)):
        print("[WARN] quality_report 缺少必要列 file/quality_score，回退为在线质量评分。")
        return {}

    qmap = {}
    for _, row in df.iterrows():
        try:
            q = float(row["quality_score"])
            qmap[_trial_key(str(row["file"]))] = float(np.clip(q, CONFIG["quality_floor"], CONFIG["quality_ceiling"]))
        except Exception:
            continue
    return qmap


def discover_latest_lag_report(save_dir: str) -> str:
    if not save_dir:
        return ""
    root = os.path.dirname(save_dir)
    pattern = os.path.join(root, "ultr_fusion_model_9_raw_v*", "all_folds_val_trial_mae.csv")
    candidates = [p for p in glob.glob(pattern) if os.path.abspath(os.path.dirname(p)) != os.path.abspath(save_dir)]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_lag_risk_map(path: str):
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] 无法读取 lag_risk_report_path: {e}")
        return {}
    required = {"trial_name", "lag_shift"}
    if not required.issubset(set(df.columns)):
        print("[WARN] lag 风险报告缺少 trial_name/lag_shift 列，跳过 lag 降权。")
        return {}
    out = {}
    for key, part in df.groupby(df["trial_name"].astype(str).map(_trial_key)):
        out[key] = float(np.mean(np.abs(part["lag_shift"].values.astype(np.float32))))
    return out


def _smooth_hybrid_ensemble(base, preds_raw: np.ndarray) -> np.ndarray:
    raw_phase = phase_from_preds(preds_raw)
    adaptive = np.asarray(smooth_predictions(base, preds_raw, strategy="adaptive_tracker"), dtype=np.float32)
    mono = np.asarray(smooth_predictions(base, preds_raw, strategy="raw_monotonic"), dtype=np.float32)
    smart = None
    if hasattr(base, "SmartGatedPhaseTracker"):
        smart = np.asarray(smooth_predictions(base, preds_raw, strategy="smart_tracker"), dtype=np.float32)

    # 置信度越高、速度越高越偏向 raw/adaptive；低置信段更多借助 mono/smart 稳定轨迹。
    vec_norm = np.sqrt(np.sum(np.square(preds_raw), axis=1)).astype(np.float32)
    conf_center = float(CONFIG.get("adaptive_conf_center", 0.85))
    conf_width = float(CONFIG.get("adaptive_conf_width", 0.30))
    conf = np.clip((vec_norm - conf_center) / max(1e-6, conf_width), 0.0, 1.0)
    vel = np.zeros_like(raw_phase, dtype=np.float32)
    if len(raw_phase) > 1:
        vel[1:] = np.abs(circular_delta(raw_phase[1:], raw_phase[:-1]))
    vel_low = float(CONFIG.get("adaptive_vel_low", 0.035))
    vel_high = float(CONFIG.get("adaptive_vel_high", 0.120))
    vel_score = np.clip((vel - vel_low) / max(1e-6, vel_high - vel_low), 0.0, 1.0)

    w_raw = 0.15 + 0.25 * conf + 0.10 * vel_score
    w_adp = 0.45 + 0.20 * vel_score
    w_mono = 0.25 + 0.20 * (1.0 - conf)
    w_smart = 0.15 + 0.15 * (1.0 - conf)
    if smart is None:
        w_smart = np.zeros_like(w_raw)

    w_sum = np.maximum(1e-6, w_raw + w_adp + w_mono + w_smart)
    w_raw, w_adp, w_mono, w_smart = w_raw / w_sum, w_adp / w_sum, w_mono / w_sum, w_smart / w_sum

    out = np.zeros_like(raw_phase, dtype=np.float32)
    for i in range(len(raw_phase)):
        local_tracks = [raw_phase[i], adaptive[i], mono[i]]
        local_weights = [w_raw[i], w_adp[i], w_mono[i]]
        if smart is not None:
            local_tracks.append(smart[i])
            local_weights.append(w_smart[i])
        out[i] = float(circular_mean_stack(local_tracks, local_weights))
    if bool(CONFIG.get("anti_spike_enable", True)):
        out = suppress_phase_spikes(
            out,
            mad_k=float(CONFIG.get("anti_spike_mad_k", 3.0)),
            abs_step_th=float(CONFIG.get("anti_spike_abs_step", 0.06)),
        )
    return out


def smooth_predictions(base, preds_raw, strategy=None):
    strategy = str(strategy or CONFIG.get("smoothing_strategy", "robust_tracker")).lower()
    preds_raw = np.asarray(preds_raw, dtype=np.float32)

    if strategy == "raw":
        return phase_from_preds(preds_raw).tolist()

    if strategy == "robust_tracker":
        tracker = base.RobustPhaseTracker(alpha=CONFIG["tracker_alpha"], beta=CONFIG["tracker_beta"])
        return [tracker.update(s, c) for s, c in preds_raw]

    if strategy == "robust_tracker_v2":
        return _smooth_robust_tracker_v2(base, preds_raw).tolist()

    if strategy == "adaptive_tracker":
        raw_phase = phase_from_preds(preds_raw)
        gated_phase = _smooth_robust_tracker_v2(base, preds_raw)
        vel = np.zeros_like(raw_phase)
        if len(raw_phase) > 1:
            vel[1:] = np.abs(circular_delta(raw_phase[1:], raw_phase[:-1]))
        vel_low = float(CONFIG.get("adaptive_vel_low", 0.035))
        vel_high = float(CONFIG.get("adaptive_vel_high", 0.120))
        vel_score = np.clip((vel - vel_low) / max(1e-6, vel_high - vel_low), 0.0, 1.0)
        vec_norm = np.sqrt(np.sum(np.square(preds_raw), axis=1))
        conf_center = float(CONFIG.get("adaptive_conf_center", 0.85))
        conf_width = float(CONFIG.get("adaptive_conf_width", 0.30))
        conf = np.clip((vec_norm - conf_center) / max(1e-6, conf_width), 0.0, 1.0)
        blend_min = float(CONFIG.get("adaptive_blend_min", 0.10))
        blend_max = float(CONFIG.get("adaptive_blend_max", 0.85))
        blend = blend_min + (blend_max - blend_min) * vel_score * (1.0 - conf)
        blend = np.clip(blend, blend_min, blend_max)
        out = (raw_phase + blend * circular_delta(gated_phase, raw_phase)) % 1.0
        if bool(CONFIG.get("anti_spike_enable", True)):
            out = suppress_phase_spikes(
                out,
                mad_k=float(CONFIG.get("anti_spike_mad_k", 3.0)),
                abs_step_th=float(CONFIG.get("anti_spike_abs_step", 0.06)),
            )
        return out.tolist()

    if strategy == "raw_monotonic":
        out = smooth_raw_monotonic(
            preds_raw,
            min_step=float(CONFIG.get("mono_min_step", -0.005)),
            max_step=float(CONFIG.get("mono_max_step", 0.080)),
        )
        if bool(CONFIG.get("anti_spike_enable", True)):
            out = suppress_phase_spikes(
                out,
                mad_k=float(CONFIG.get("anti_spike_mad_k", 3.0)),
                abs_step_th=float(CONFIG.get("anti_spike_abs_step", 0.06)),
            )
        return out.tolist()

    if strategy == "lowpass":
        smoother = base.EnhancedVectorSmoother(alpha=CONFIG["ema_alpha"])
        base_phases = [smoother.update(s, c) for s, c in preds_raw]
        unwrapped_phase = np.unwrap(np.array(base_phases) * 2 * np.pi)
        b, a = butter(N=CONFIG["butter_order"], Wn=CONFIG["butter_cutoff"], btype="low")
        smoothed = filtfilt(b, a, unwrapped_phase)
        return ((smoothed / (2 * np.pi)) % 1.0).tolist()

    if strategy == "smart_tracker":
        if not hasattr(base, "SmartGatedPhaseTracker"):
            raise AttributeError("base 脚本缺少 SmartGatedPhaseTracker，无法使用 smart_tracker。")
        tracker = base.SmartGatedPhaseTracker()
        return [tracker.update(s, c) for s, c in preds_raw]

    if strategy == "hybrid_ensemble":
        return _smooth_hybrid_ensemble(base, preds_raw).tolist()

    smoother = base.EnhancedVectorSmoother(alpha=CONFIG["ema_alpha"])
    return [smoother.update(s, c) for s, c in preds_raw]


def resolve_strategy_candidates(base):
    known = {
        "raw",
        "robust_tracker",
        "robust_tracker_v2",
        "adaptive_tracker",
        "raw_monotonic",
        "lowpass",
        "smart_tracker",
        "hybrid_ensemble",
        "basic",
    }
    raw_candidates = CONFIG.get("strategy_candidates", [CONFIG.get("smoothing_strategy", "adaptive_tracker")])
    if isinstance(raw_candidates, str):
        raw_candidates = [raw_candidates]

    out = []
    for s in raw_candidates:
        key = str(s).lower().strip()
        if not key:
            continue
        if key not in known:
            print(f"[StrategyEval] 跳过未知策略: {key}")
            continue
        if key == "smart_tracker" and not hasattr(base, "SmartGatedPhaseTracker"):
            print("[StrategyEval] 跳过 smart_tracker（base 无 SmartGatedPhaseTracker）。")
            continue
        if key == "basic":
            key = "robust_tracker"
        if key not in out:
            out.append(key)
    if not out:
        out = [str(CONFIG.get("smoothing_strategy", "adaptive_tracker")).lower()]
    return out


def evaluate_one_strategy(base, preds_raw, true_phases, strategy, strict_eval, primary_mode):
    pred_phases = np.asarray(smooth_predictions(base, preds_raw, strategy=strategy), dtype=np.float32)
    mae_no_lag = circular_mae(pred_phases, true_phases)
    lag_shift = 0
    pred_oracle = pred_phases
    mae_oracle = mae_no_lag
    if bool(CONFIG.get("lag_comp_enable", True)):
        pred_oracle, lag_shift, mae_after_lag = best_lag_compensation(
            pred_phases, true_phases, int(CONFIG.get("lag_comp_max_shift", 6))
        )
        mae_oracle = mae_after_lag

    use_no_lag_primary = strict_eval and primary_mode == "no_lag"
    pred_eval = pred_phases if use_no_lag_primary else pred_oracle
    trial_mae = mae_no_lag if use_no_lag_primary else mae_oracle
    jr = jump_rate(pred_phases, float(CONFIG.get("diag_jump_threshold", 0.08)))
    sr = spike_rate(pred_phases, float(CONFIG.get("diag_spike_threshold", 0.06)))
    edge_ratio_eval = dynamic_edge_ratio_from_phase(true_phases)
    e_mae = edge_phase_mae(pred_eval, true_phases, edge_ratio_eval)
    s_mae = slope_phase_mae(pred_eval, true_phases)

    edge_w = float(CONFIG.get("strategy_score_edge_weight", 0.15))
    jump_w = float(CONFIG.get("strategy_score_jump_weight", 0.05))
    spike_w = float(CONFIG.get("strategy_score_spike_weight", 0.05))
    slope_w = float(CONFIG.get("strategy_score_slope_weight", 0.08))
    e_mae_safe = float(trial_mae if not np.isfinite(e_mae) else e_mae)
    s_mae_safe = float(0.0 if not np.isfinite(s_mae) else s_mae)
    composite = float(trial_mae + edge_w * e_mae_safe + jump_w * jr + spike_w * sr + slope_w * s_mae_safe)

    return {
        "strategy": strategy,
        "pred_phases": pred_phases,
        "pred_oracle": np.asarray(pred_oracle, dtype=np.float32),
        "trial_mae": float(trial_mae),
        "trial_mae_no_lag": float(mae_no_lag),
        "trial_mae_oracle_lag": float(mae_oracle),
        "lag_shift": int(lag_shift),
        "jump_rate": float(jr),
        "spike_ratio": float(sr),
        "edge_mae": float(e_mae),
        "slope_mae": float(s_mae),
        "composite_score": composite,
    }



def main():
    apply_env_overrides()
    np.random.seed(CONFIG.get("random_seed", 42))
    tf.random.set_seed(CONFIG.get("random_seed", 42))
    base = load_base_module(BASE_TRAIN_SCRIPT)
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])

    print("1. 加载并评分所有 trial ...")
    raw_trials = base.load_all_data_without_split(CONFIG["data_dir"])
    if not raw_trials:
        raise ValueError("未找到可训练数据。")

    external_qmap = load_external_quality_map(CONFIG.get("quality_report_path", ""))
    if external_qmap:
        print(f"已加载外部质量报告: {CONFIG['quality_report_path']} (entries={len(external_qmap)})")
    else:
        print("未加载外部质量报告，将使用在线质量评分。")

    scored_trials = []
    dropped_short_rows = []
    report_lines = [
        "name,subject,domain,n_samples,quality,quality_source,saturation_ratio,phase_coverage,effective_cycles"
    ]
    for name, feat, label in raw_trials:
        if CONFIG.get("drop_short_trials_enable", False) and len(label) < int(CONFIG.get("drop_short_trials_frames", 95)):
            dropped_short_rows.append(
                {
                    "name": name,
                    "n_samples": int(len(label)),
                    "drop_rule": f"short_clip<{int(CONFIG.get('drop_short_trials_frames', 95))}",
                }
            )
            continue

        subject = parse_subject_id(name)
        domain = parse_domain(name)
        q_online, sat, cov, cyc = trial_quality_score(feat, label)
        q_external = external_qmap.get(_trial_key(name))
        if q_external is None:
            q = q_online
            q_source = "online"
        else:
            q = q_external
            q_source = "external_report"
        scored_trials.append((name, feat, label, subject, domain, q, sat, cov, cyc))
        report_lines.append(
            f"{name},{subject},{domain},{len(label)},{q:.4f},{q_source},{sat:.4f},{cov:.4f},{cyc:.4f}"
        )

    if dropped_short_rows:
        dropped_df = pd.DataFrame(dropped_short_rows)
        dropped_path = os.path.join(CONFIG["save_dir"], "dropped_short_trials.csv")
        dropped_df.to_csv(dropped_path, index=False, encoding="utf-8-sig")
        print(f"[规则] 短序列硬剔除: {len(dropped_short_rows)} 条，详见 {dropped_path}")
    with open(os.path.join(CONFIG["save_dir"], "trial_quality_report.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"共 {len(scored_trials)} 条 trial，已输出质量报告。")
    lag_risk_path = str(CONFIG.get("lag_risk_report_path", "")).strip()
    lag_penalty_flag = bool(CONFIG.get("lag_penalty_enable", True))
    if bool(CONFIG.get("strict_eval_enable", True)) and bool(CONFIG.get("strict_disable_external_lag_penalty", True)):
        lag_penalty_flag = False
        print("[StrictEval] 已关闭 lag_penalty（避免外部先验影响外层验证严谨性）。")
    if not lag_risk_path and lag_penalty_flag:
        lag_risk_path = discover_latest_lag_report(CONFIG["save_dir"])
        if lag_risk_path:
            print(f"[LagPenalty] 自动发现历史报告: {lag_risk_path}")
    lag_risk_map = load_lag_risk_map(lag_risk_path) if lag_penalty_flag else {}
    if lag_risk_map:
        print(f"[LagPenalty] 已加载 lag 风险映射: {len(lag_risk_map)} 条")

    groups = [t[3] for t in scored_trials]
    unique_group_count = len(set(groups))
    n_trials = len(scored_trials)
    target_folds = CONFIG["k_folds"]
    if n_trials < 15 and target_folds > 3:
        target_folds = 3
        print(f"[WARN] 当前 trial 数较少 ({n_trials})，K 折从 {CONFIG['k_folds']} 自动下调到 {target_folds}。")
    n_splits = min(target_folds, unique_group_count)
    if n_splits < 2:
        raise ValueError(f"可用受试者组数量不足，无法做分组交叉验证。groups={unique_group_count}")

    cv_strategy = str(CONFIG.get("cv_strategy", "balanced_group")).lower()
    if cv_strategy == "groupkfold":
        gkf = GroupKFold(n_splits=n_splits)
        fold_splits = list(gkf.split(np.zeros(len(scored_trials)), groups=groups))
        print(f"[CV] strategy=groupkfold, n_splits={n_splits}")
    else:
        fold_splits, fold_groups, fold_sizes = build_balanced_group_splits(
            groups=groups,
            n_splits=n_splits,
            random_seed=int(CONFIG.get("cv_random_seed", CONFIG.get("random_seed", 42))),
            shuffle_groups=bool(CONFIG.get("cv_shuffle_groups", True)),
        )
        print(f"[CV] strategy=balanced_group, n_splits={len(fold_splits)}, fold_sizes={fold_sizes}")
        print(f"[CV] fold_groups={ [sorted(list(gs)) for gs in fold_groups] }")

    total_folds = len(fold_splits)
    strategy_eval_enable = bool(CONFIG.get("strategy_eval_enable", True))
    strategy_candidates = resolve_strategy_candidates(base)
    print(f"[StrategyEval] enable={strategy_eval_enable}, candidates={strategy_candidates}")
    calib_keys = [
        "mono_max_step",
        "mono_min_step",
        "diag_jump_threshold",
        "diag_spike_threshold",
        "lag_comp_max_shift",
    ]
    calib_base = {k: CONFIG.get(k) for k in calib_keys}
    all_fold_maes = []
    all_fold_trial_details = []

    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n================ 开始训练第 {fold}/{total_folds} 折 ================")
        train_trials = [scored_trials[i] for i in train_idx]
        val_trials = [scored_trials[i] for i in val_idx]
        for k in calib_keys:
            CONFIG[k] = calib_base[k]
        auto_calibrate_postprocess(train_trials, tag=f"fold_{fold}")

        domain_counter = Counter([t[4] for t in train_trials])
        n_domains = len(domain_counter)
        total_trials = len(train_trials)
        domain_weight_map = {}
        manual_overrides = bool(CONFIG.get("manual_weight_overrides_enable", False))
        for d, c in domain_counter.items():
            if CONFIG["domain_balance_enable"]:
                domain_weight_map[d] = total_trials / max(1.0, n_domains * c)
            else:
                domain_weight_map[d] = 1.0
            if manual_overrides:
                domain_weight_map[d] *= float(CONFIG.get("domain_weight_overrides", {}).get(d, 1.0))
            domain_weight_map[d] = float(
                np.clip(
                    domain_weight_map[d],
                    CONFIG.get("domain_weight_min", 0.75),
                    CONFIG.get("domain_weight_max", 1.20),
                )
            )

        aug_train_features, aug_train_labels = [], []
        all_train_feat_concat = []
        train_meta = []  # (quality_weight * domain_weight)

        lag_penalized_count = 0
        apply_lag_penalty = bool(lag_risk_map)
        risky_train_keys = set()
        if apply_lag_penalty:
            for trial_name, *_ in train_trials:
                lag_abs = lag_risk_map.get(_trial_key(trial_name))
                if lag_abs is not None and lag_abs >= float(CONFIG.get("lag_penalty_abs_shift", 4)):
                    risky_train_keys.add(_trial_key(trial_name))
            risky_ratio = len(risky_train_keys) / max(1, len(train_trials))
            if risky_ratio > float(CONFIG.get("lag_penalty_max_ratio", 0.35)):
                apply_lag_penalty = False
                print(
                    f"[LagPenalty] fold {fold}: risky_ratio={risky_ratio:.2f} 超过阈值，"
                    "本折关闭 lag 降权以避免欠拟合。"
                )

        for trial_name, feat, label, subject, domain, quality, *_ in train_trials:
            domain_w = domain_weight_map.get(domain, 1.0)
            if manual_overrides:
                subject_w = CONFIG["subject_weight_map"].get(str(subject), CONFIG["subject_weight_default"])
                subject_w = float(np.clip(subject_w, CONFIG["subject_weight_min"], CONFIG["subject_weight_max"]))
            else:
                subject_w = 1.0
            lag_w = 1.0
            if apply_lag_penalty:
                if _trial_key(trial_name) in risky_train_keys:
                    lag_w = float(CONFIG.get("lag_penalty_weight", 0.85))
                    lag_penalized_count += 1
            # Allow lower weights for low-quality trials to reduce noisy supervision.
            trial_w = float(
                np.clip(
                    quality * domain_w * subject_w * lag_w,
                    CONFIG.get("sample_weight_min", 0.70),
                    CONFIG.get("sample_weight_max", 1.30),
                )
            )
            for factor in CONFIG["augment_factors"]:
                f_aug, l_aug = base.augment_time_warp(feat, label, factor)
                aug_train_features.append(f_aug)
                aug_train_labels.append(l_aug)
                all_train_feat_concat.append(f_aug)
                train_meta.append(trial_w)

        all_train_feat_concat = np.vstack(all_train_feat_concat)
        scaler = AdaptiveScaler(mode=CONFIG.get("normalization_mode", "global"))
        if CONFIG.get("normalization_mode", "global") == "global":
            all_train_feat_scaled = scaler.fit_transform(all_train_feat_concat)
        else:
            all_train_feat_scaled = np.vstack([scaler.transform(f) for f in aug_train_features])
        pca = PCA(n_components=CONFIG["pca_components"])
        pca.fit(all_train_feat_scaled)
        joblib.dump(scaler, os.path.join(CONFIG["save_dir"], f"scaler_fold_{fold}.pkl"))
        joblib.dump(pca, os.path.join(CONFIG["save_dir"], f"pca_fold_{fold}.pkl"))

        X_train_seqs, y_train_seqs, train_weights = [], [], []
        for feat, label, trial_w in zip(aug_train_features, aug_train_labels, train_meta):
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = base.create_sequences(f_pca, label, CONFIG["sequence_length"])
            if len(X_seq) == 0:
                continue
            X_train_seqs.append(X_seq)
            y_train_seqs.append(y_seq)

            # [修改点 4] 彻底修复边界加权逻辑：根据真实的相边界加权，而不是序列收尾
            if CONFIG["edge_weight_enable"]:
                # 还原这段序列的真实相位
                true_rad = np.arctan2(y_seq[:, 0], y_seq[:, 1])
                true_phases = (true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
                edge_ratio_local = dynamic_edge_ratio_from_phase(true_phases)

                # 初始化全1权重
                w_seq = np.ones(len(y_seq), dtype=np.float32)

                # 找到相位在 [0, ratio] 或者[1-ratio, 1.0] 的跳变边界区域
                edge_mask = (true_phases < edge_ratio_local) | (true_phases > (1.0 - edge_ratio_local))

                # 对跳变区域赋予高权重惩罚
                w_seq[edge_mask] = CONFIG["edge_weight"]
            else:
                w_seq = np.ones(len(X_seq), dtype=np.float32)

            train_weights.append((w_seq * trial_w).astype(np.float32))
        if lag_risk_map:
            print(f"[LagPenalty] fold {fold}: penalized_train_trials={lag_penalized_count}/{len(train_trials)}")

        X_train, y_train = np.vstack(X_train_seqs), np.vstack(y_train_seqs)
        sample_weights = np.concatenate(train_weights).astype(np.float32)
        idx = np.random.permutation(len(X_train))
        X_train, y_train, sample_weights = X_train[idx], y_train[idx], sample_weights[idx]

        X_val_seqs, y_val_seqs = [], []
        for _, feat, label, *_ in val_trials:
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = base.create_sequences(f_pca, label, CONFIG["sequence_length"])
            if len(X_seq) == 0:
                continue
            X_val_seqs.append(X_seq)
            y_val_seqs.append(y_seq)
        if not X_val_seqs:
            print(f"[WARN] 第 {fold} 折验证集没有可用序列，跳过该折。")
            continue
        X_val, y_val = np.vstack(X_val_seqs), np.vstack(y_val_seqs)

        base.CONFIG["learning_rate"] = CONFIG["learning_rate"]
        base.CONFIG["l2_reg"] = base.CONFIG.get("l2_reg", 0.001)
        model = base.build_fusion_model(input_shape=(CONFIG["sequence_length"], CONFIG["pca_components"]))

        callbacks = [
            ModelCheckpoint(
                os.path.join(CONFIG["save_dir"], f"best_model_fold_{fold}.keras"),
                monitor="val_phase_mae",
                save_best_only=True,
                mode="min",
            ),
            ReduceLROnPlateau(
                monitor="val_phase_mae",
                factor=CONFIG["reduce_lr_factor"],
                patience=CONFIG["reduce_lr_patience"],
                verbose=0,
                mode="min",
            ),
            EarlyStopping(
                monitor="val_phase_mae",
                patience=CONFIG["early_stop_patience"],
                restore_best_weights=True,
                verbose=0,
                mode="min",
            ),
        ]

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            validation_data=(X_val, y_val),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"],
            callbacks=callbacks,
            verbose=2,
        )

        print(f"\n正在评估第 {fold} 折验证集...")
        fold_trial_maes = []
        fold_trial_details = []
        fold_trial_plot_cache = {}
        strict_eval = bool(CONFIG.get("strict_eval_enable", True))
        primary_mode = str(CONFIG.get("strict_eval_primary", "no_lag")).lower()
        val_cache = []
        for trial_name, test_feat, test_label, subject, domain, quality, sat, cov, cyc in val_trials:
            f_scaled = scaler.transform(test_feat)
            f_pca = pca.transform(f_scaled)
            X_test_trial, y_test_trial = base.create_sequences(f_pca, test_label, CONFIG["sequence_length"])
            if len(X_test_trial) == 0:
                continue
            preds_raw = model.predict(X_test_trial, verbose=0)
            true_rad = np.arctan2(y_test_trial[:, 0], y_test_trial[:, 1])
            true_phases = np.asarray((true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi), dtype=np.float32)
            val_cache.append(
                {
                    "trial_name": trial_name,
                    "subject": subject,
                    "domain": domain,
                    "quality": float(quality),
                    "sat": float(sat),
                    "cov": float(cov),
                    "cyc": float(cyc),
                    "n_raw_frames": int(len(test_label)),
                    "n_eval_sequences": int(len(X_test_trial)),
                    "preds_raw": np.asarray(preds_raw, dtype=np.float32),
                    "true_phases": true_phases,
                }
            )

        if not val_cache:
            print(f"[WARN] 第 {fold} 折无可评估 trial，跳过该折。")
            continue

        active_candidates = (
            strategy_candidates
            if strategy_eval_enable
            else [str(CONFIG.get("smoothing_strategy", "adaptive_tracker")).lower()]
        )
        strategy_rows = []
        for strategy in active_candidates:
            trial_scores = []
            trial_mae_no_lag = []
            trial_mae_primary = []
            trial_edge = []
            trial_jump = []
            trial_spike = []
            trial_slope = []
            for item in val_cache:
                metrics = evaluate_one_strategy(
                    base=base,
                    preds_raw=item["preds_raw"],
                    true_phases=item["true_phases"],
                    strategy=strategy,
                    strict_eval=strict_eval,
                    primary_mode=primary_mode,
                )
                trial_scores.append(metrics["composite_score"])
                trial_mae_no_lag.append(metrics["trial_mae_no_lag"])
                trial_mae_primary.append(metrics["trial_mae"])
                trial_edge.append(metrics["edge_mae"])
                trial_jump.append(metrics["jump_rate"])
                trial_spike.append(metrics["spike_ratio"])
                trial_slope.append(metrics["slope_mae"])
            strategy_rows.append(
                {
                    "fold": fold,
                    "strategy": strategy,
                    "mean_composite_score": float(np.mean(trial_scores)),
                    "mean_trial_mae_primary": float(np.mean(trial_mae_primary)),
                    "mean_trial_mae_no_lag": float(np.mean(trial_mae_no_lag)),
                    "mean_edge_mae": float(np.nanmean(np.asarray(trial_edge, dtype=np.float32))),
                    "mean_jump_rate": float(np.mean(trial_jump)),
                    "mean_spike_ratio": float(np.mean(trial_spike)),
                    "mean_slope_mae": float(np.nanmean(np.asarray(trial_slope, dtype=np.float32))),
                    "n_trials": int(len(trial_scores)),
                }
            )

        strategy_df = pd.DataFrame(strategy_rows).sort_values("mean_composite_score", ascending=True)
        strategy_path = os.path.join(CONFIG["save_dir"], f"fold_{fold}_strategy_benchmark.csv")
        strategy_df.to_csv(strategy_path, index=False, encoding="utf-8-sig")
        best_strategy = str(strategy_df.iloc[0]["strategy"])
        print(
            f"[StrategyEval] fold {fold} 最优策略: {best_strategy} | "
            f"score={float(strategy_df.iloc[0]['mean_composite_score']):.5f}, "
            f"MAE_primary={float(strategy_df.iloc[0]['mean_trial_mae_primary']) * 100:.2f}%"
        )

        for item in val_cache:
            metrics = evaluate_one_strategy(
                base=base,
                preds_raw=item["preds_raw"],
                true_phases=item["true_phases"],
                strategy=best_strategy,
                strict_eval=strict_eval,
                primary_mode=primary_mode,
            )
            pred_phases = metrics["pred_phases"]
            pred_oracle = metrics["pred_oracle"]
            trial_mae = metrics["trial_mae"]
            mae_no_lag = metrics["trial_mae_no_lag"]
            mae_oracle = metrics["trial_mae_oracle_lag"]
            lag_shift = metrics["lag_shift"]
            jr = metrics["jump_rate"]
            sr = metrics["spike_ratio"]
            e_mae = metrics["edge_mae"]
            s_mae = metrics["slope_mae"]
            fold_trial_maes.append(trial_mae)
            fold_trial_details.append(
                {
                    "fold": fold,
                    "trial_name": item["trial_name"],
                    "subject": item["subject"],
                    "domain": item["domain"],
                    "strategy": best_strategy,
                    "strategy_composite_score": float(metrics["composite_score"]),
                    "quality_weight": float(item["quality"]),
                    "saturation_ratio": float(item["sat"]),
                    "phase_coverage": float(item["cov"]),
                    "effective_cycles": float(item["cyc"]),
                    "n_raw_frames": int(item["n_raw_frames"]),
                    "n_eval_sequences": int(item["n_eval_sequences"]),
                    "trial_mae": float(trial_mae),
                    "trial_mae_before_lag": float(mae_no_lag),
                    "trial_mae_no_lag": float(mae_no_lag),
                    "trial_mae_oracle_lag": float(mae_oracle),
                    "lag_shift": int(lag_shift),
                    "jump_rate": float(jr),
                    "spike_ratio": float(sr),
                    "edge_mae": float(e_mae),
                    "slope_mae": float(s_mae),
                }
            )
            plot_phase = pred_oracle if bool(CONFIG.get("plot_use_lag_comp_phase", False)) else pred_phases
            fold_trial_plot_cache[item["trial_name"]] = {
                "true_phases": item["true_phases"].copy(),
                "plot_phase": np.asarray(plot_phase, dtype=np.float32).copy(),
                "trial_mae": float(trial_mae),
                "strategy": best_strategy,
            }
        current_fold_mae = float(np.mean(fold_trial_maes))
        all_fold_maes.append(current_fold_mae)
        metric_tag = "no_lag" if (strict_eval and primary_mode == "no_lag") else "oracle_lag"
        print(f"第 {fold} 折验证集平均 MAE[{metric_tag}]: {current_fold_mae * 100:.2f}% (strategy={best_strategy})")

        fold_df = pd.DataFrame(fold_trial_details).sort_values("trial_mae", ascending=False)
        all_fold_trial_details.extend(fold_trial_details)
        fold_report_path = os.path.join(CONFIG["save_dir"], f"fold_{fold}_val_trial_mae.csv")
        fold_bad_top_path = os.path.join(CONFIG["save_dir"], f"fold_{fold}_bad_trials_top10.csv")
        fold_df.to_csv(fold_report_path, index=False, encoding="utf-8-sig")
        fold_df.head(10).to_csv(fold_bad_top_path, index=False, encoding="utf-8-sig")
        print(f"[诊断] 已保存第 {fold} 折 trial 级报告: {fold_report_path}")
        if len(fold_df):
            top1 = fold_df.iloc[0]
            print(
                f"[诊断] 第 {fold} 折最差样本: {top1['trial_name']} "
                f"(MAE={top1['trial_mae'] * 100:.2f}%, Q={top1['quality_weight']:.3f}, "
                f"Sat={top1['saturation_ratio']:.3f}, Cov={top1['phase_coverage']:.3f})"
            )

        # Save representative trial plots: worst / median / best.
        ranked_trials = fold_df["trial_name"].tolist()
        rep_index_map = {
            "worst": 0,
            "median": len(ranked_trials) // 2,
            "best": len(ranked_trials) - 1,
        }
        plotted = []
        used_trial = set()
        for tag, ridx in rep_index_map.items():
            if not ranked_trials:
                continue
            trial_name = ranked_trials[ridx]
            if trial_name in used_trial:
                continue
            cache = fold_trial_plot_cache.get(trial_name)
            if cache is None:
                continue
            used_trial.add(trial_name)
            plotted.append((tag, trial_name, cache["trial_mae"]))
            true_curve = cache["true_phases"]
            pred_curve = cache["plot_phase"]
            limit = min(500, len(true_curve))
            plt.figure(figsize=(15, 6))
            plt.plot(true_curve[:limit], "k-", alpha=0.5, linewidth=3, label="Ground Truth Phase")
            plt.plot(
                pred_curve[:limit],
                "r--",
                linewidth=2,
                label=f"Predicted Phase ({cache.get('strategy', best_strategy)})",
            )
            plt.title(
                f"Fold {fold} {tag.capitalize()} Trial - MAE: {cache['trial_mae'] * 100:.2f}% "
                f"(plot_lag_comp={bool(CONFIG.get('plot_use_lag_comp_phase', False))})"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["save_dir"], f"fold_{fold}_{tag}_trial.png"))
            plt.close()

        # Keep backward-compatible output name.
        if plotted:
            _, median_name, _ = next((x for x in plotted if x[0] == "median"), plotted[0])
            cache = fold_trial_plot_cache.get(median_name)
            if cache is not None:
                true_curve = cache["true_phases"]
                pred_curve = cache["plot_phase"]
                limit = min(500, len(true_curve))
                plt.figure(figsize=(15, 6))
                plt.plot(true_curve[:limit], "k-", alpha=0.5, linewidth=3, label="Ground Truth Phase")
                plt.plot(
                    pred_curve[:limit],
                    "r--",
                    linewidth=2,
                    label=f"Predicted Phase ({cache.get('strategy', best_strategy)})",
                )
                plt.title(
                    f"Fold {fold} Median Trial - MAE: {cache['trial_mae'] * 100:.2f}% "
                    f"(plot_lag_comp={bool(CONFIG.get('plot_use_lag_comp_phase', False))})"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(CONFIG["save_dir"], f"fold_{fold}_result.png"))
                plt.close()
            print(f"[诊断] 第 {fold} 折代表样本图: {plotted}")

        tf.keras.backend.clear_session()
        best_fold_idx = int(np.argmin(all_fold_maes)) + 1
        shutil.copy(
            os.path.join(CONFIG["save_dir"], f"best_model_fold_{best_fold_idx}.keras"),
            os.path.join(CONFIG["save_dir"], "best_model_overall.keras"),
        )
        shutil.copy(
            os.path.join(CONFIG["save_dir"], f"scaler_fold_{best_fold_idx}.pkl"),
            os.path.join(CONFIG["save_dir"], "best_scaler_overall.pkl"),
        )
        shutil.copy(
            os.path.join(CONFIG["save_dir"], f"pca_fold_{best_fold_idx}.pkl"),
            os.path.join(CONFIG["save_dir"], "best_pca_overall.pkl"),
        )

    if all_fold_trial_details:
        all_trial_df = pd.DataFrame(all_fold_trial_details)
        all_trial_path = os.path.join(CONFIG["save_dir"], "all_folds_val_trial_mae.csv")
        all_trial_df.sort_values(["fold", "trial_mae"], ascending=[True, False]).to_csv(
            all_trial_path, index=False, encoding="utf-8-sig"
        )
        worst_path = os.path.join(CONFIG["save_dir"], "all_folds_worst_trials_top30.csv")
        all_trial_df.sort_values("trial_mae", ascending=False).head(30).to_csv(
            worst_path, index=False, encoding="utf-8-sig"
        )
        subject_summary = (
            all_trial_df.groupby("subject")
            .agg(
                n_trials=("trial_mae", "size"),
                mean_trial_mae=("trial_mae", "mean"),
                mean_trial_mae_before_lag=("trial_mae_before_lag", "mean"),
                mean_abs_lag=("lag_shift", lambda s: float(np.mean(np.abs(s)))),
                mean_edge_mae=("edge_mae", "mean"),
                mean_jump_rate=("jump_rate", "mean"),
                mean_spike_ratio=("spike_ratio", "mean"),
            )
            .reset_index()
            .sort_values("mean_trial_mae", ascending=False)
        )
        subject_path = os.path.join(CONFIG["save_dir"], "all_folds_subject_risk_summary.csv")
        subject_summary.to_csv(subject_path, index=False, encoding="utf-8-sig")
        print(f"[诊断] 已保存跨折 trial 级汇总: {all_trial_path}")
        print(f"[诊断] 已保存跨折最差样本 Top30: {worst_path}")
        print(f"[诊断] 已保存受试者风险汇总: {subject_path}")

    strategy_files = sorted(glob.glob(os.path.join(CONFIG["save_dir"], "fold_*_strategy_benchmark.csv")))
    if strategy_files:
        all_rows = []
        for p in strategy_files:
            try:
                all_rows.append(pd.read_csv(p))
            except Exception:
                continue
        if all_rows:
            merged = pd.concat(all_rows, ignore_index=True)
            global_rank = (
                merged.groupby("strategy")
                .agg(
                    n_folds=("fold", "nunique"),
                    mean_composite_score=("mean_composite_score", "mean"),
                    mean_trial_mae_primary=("mean_trial_mae_primary", "mean"),
                    mean_trial_mae_no_lag=("mean_trial_mae_no_lag", "mean"),
                    mean_edge_mae=("mean_edge_mae", "mean"),
                    mean_jump_rate=("mean_jump_rate", "mean"),
                    mean_spike_ratio=("mean_spike_ratio", "mean"),
                    mean_slope_mae=("mean_slope_mae", "mean"),
                )
                .reset_index()
                .sort_values("mean_composite_score", ascending=True)
            )
            global_path = os.path.join(CONFIG["save_dir"], "all_folds_strategy_benchmark.csv")
            global_rank.to_csv(global_path, index=False, encoding="utf-8-sig")
            if len(global_rank):
                winner = global_rank.iloc[0]
                print(
                    f"[StrategyEval] 全折最优策略: {winner['strategy']} | "
                    f"score={float(winner['mean_composite_score']):.5f}, "
                    f"MAE_primary={float(winner['mean_trial_mae_primary']) * 100:.2f}%"
                )
                print(f"[StrategyEval] 全折策略榜单已保存: {global_path}")

    print("================================================================")
    print(f"Model 9 全部完成，平均 MAE: {np.mean(all_fold_maes) * 100:.2f}%")
    print("================================================================")


if __name__ == "__main__":
    main()