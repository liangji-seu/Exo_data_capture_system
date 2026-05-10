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
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from scipy.signal import butter, filtfilt, medfilt

plt.rcParams["font.sans-serif"] =["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
# ============================== 0. 全局中控台 (CONFIG) ========================
# ==============================================================================
CONFIG = {
    # 这份脚本做的事情可以概括为：
    # - 调用 `base_train_script` 里的数据加载逻辑拿到所有 trial；
    # - 对每个 trial 计算“数据质量分”并可选启用滞后惩罚/严谨评估；
    # - 对预测后处理的多种平滑策略做折内/折外评估，最终用于确定“最优后处理策略”。

    # ---------------- [模块一] 路径与环境配置 ----------------
    "base_train_script": r"D:\nightmare\Documents\SRTP\LSTM\模型9_增强.py",  # 原双向LSTM脚本路径
    "data_dir": r"D:\nightmare\Documents\SRTP\AllData\Merge_Data_Process\pipeline_run_20260319_220903\04_clean_dataset",
    "save_dir": r"D:\nightmare\Documents\SRTP\LSTM\training_logs\ultr_fusion_model_9_auto_pro_2",
    "quality_report_path": r"D:\nightmare\Documents\SRTP\AllData\Merge_Data_Process\pipeline_run_20260319_220903\03_reports\clean_set_build\clean_build_report.csv",
    "lag_risk_report_path": "",
    
    # ---------------- [模块二] 训练参数与交叉验证 ----------------
    "k_folds": 5,
    "cv_strategy": "balanced_group",       # 切分策略: balanced_group | groupkfold
    "cv_shuffle_groups": True,
    "cv_random_seed": 42,
    "random_seed": 42,
    "sequence_length": 48,                 # 感受野
    "pca_components": 20,
    "normalization_mode": "trial",         # global | trial
    "augment_factors":[0.8, 0.9, 1.0, 1.1],
    "epochs": 150,
    "batch_size": 32,
    "learning_rate": 0.0015,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 6,
    "early_stop_patience": 25,

    # ---------------- [模块三] 边界加权控制 ----------------
    "edge_weight_enable": True,            # [开关] 是否加权惩罚边界
    "edge_ratio": 0.15,                    # 相位[0, 0.15] & [0.85, 1.0]
    "edge_weight": 3.5,                    # 边界权重倍率
    "edge_dynamic_enable": False,
    "edge_target_frames": 10,
    "edge_ratio_min": 0.08,
    "edge_ratio_max": 0.24,

    # ----------------[模块四] 数据质量与自动降权机制 ----------------
    "sensor_low_clip": 55.0,
    "sensor_high_clip": 205.0,
    "sat_warn_threshold": 0.08,
    "sat_bad_threshold": 0.22,
    "phase_coverage_min": 0.82,
    "effective_cycles_min": 4.0,
    "quality_floor": 0.70,
    "quality_ceiling": 1.10,

    # 切片过滤与受试者映射降权
    "drop_short_trials_enable": True,
    "drop_short_trials_frames": 110,
    "short_clip_warn_frames": 120,
    "short_clip_drop_frames": 95,
    "short_clip_penalty_weight": 0.55,
    "manual_weight_overrides_enable": True,
    "domain_balance_enable": False,
    "domain_weight_min": 0.75,
    "domain_weight_max": 1.20,
    "domain_weight_overrides": {"Processed_Final_Training_Set_1.14_1": 0.85},
    "subject_weight_map": {"11": 0.85, "15": 0.80, "18": 0.92, "23": 0.90},
    "subject_weight_default": 1.00,
    "subject_weight_min": 0.80,
    "subject_weight_max": 1.50,
    "sample_weight_min": 0.70,
    "sample_weight_max": 1.30,

    # ---------------- [模块五] 滞后惩罚与参数自校准 ----------------
    "lag_comp_enable": True,
    "lag_comp_max_shift": 6,
    "lag_penalty_enable": False,
    "lag_penalty_abs_shift": 4,
    "lag_penalty_weight": 0.85,
    "lag_penalty_max_ratio": 0.35,

    "auto_calibrate_postprocess": False,
    "auto_calibrate_verbose": True,
    "auto_calibrate_mode": "guarded",      # guarded | legacy
    "auto_calibrate_min_trials": 24,
    "auto_calibrate_cycle_cv_max": 0.22,
    "auto_calibrate_max_rel_change": 0.20,
    
    # 诊断参数
    "diag_jump_threshold": 0.08,
    "diag_spike_threshold": 0.06,

    # ----------------[模块六] 后处理平滑策略评估池 ----------------
    "smoothing_strategy": "perfect_sawtooth", # 默认与基础展示策略
    "strategy_eval_enable": True,             # [开关] 自动多策略评估
    "strategy_candidates":[
        "adaptive_tracker", "robust_tracker_v2", "smart_tracker",
        "raw_monotonic", "robust_tracker", "lowpass", "raw", "hybrid_ensemble",
        "perfect_sawtooth", "anchor_sync"     # 终极积分器与锚点插值
    ],

    # 打分权重机制 (越低越好)
    # 折内选策略时使用的聚合方式
    # - mean_composite: 兼容旧逻辑（使用 mean(composite_score)）
    # - median_mae_std: fold_score = median(trial_mae_primary) + std_weight * std(trial_mae_primary)
    "strategy_fold_score_mode": "mean_composite", 
    "strategy_fold_score_std_weight": 0.5,
    "strategy_score_edge_weight": 0.15,
    "strategy_score_jump_weight": 0.05,
    "strategy_score_spike_weight": 0.05,
    "strategy_score_slope_weight": 0.08,

    # 平滑与防刺底层参数
    "tracker_alpha": 0.75,
    "tracker_beta": 0.03,
    "tracker_max_step": 0.12,
    "adaptive_vel_low": 0.035,
    "adaptive_vel_high": 0.120,
    "adaptive_blend_min": 0.10,
    "adaptive_blend_max": 0.85,
    "adaptive_conf_center": 0.85,
    "adaptive_conf_width": 0.30,
    "ema_alpha": 0.6,
    "butter_order": 3,
    "butter_cutoff": 0.1,
    "mono_min_step": -0.005,
    "mono_max_step": 0.080,
    "wrap_hysteresis_enable": False,
    "wrap_arm_high": 0.90,
    "wrap_release_low": 0.10,
    "anti_spike_enable": False,
    "anti_spike_mad_k": 3.0,
    "anti_spike_abs_step": 0.06,

    # ---------------- [模块七] 验证与评估严谨性控制 ----------------
    # 实验开关：将严格评估主指标切换到 oracle lag（对齐后的 MAE），并默认在画图时使用滞后对齐曲线。
    # 需要回退时，把它改回 False（脚本会回到原来的 strict_eval_primary 与 plot_use_lag_comp_phase 配置）。
    "causal_strict_enable": False,            # [开关] 是否开启绝对因果限制
    "strict_disable_i1_usage_enable": False,  
    "strict_eval_enable": True,
    "strict_eval_primary": "no_lag",          # no_lag | oracle_lag
    "strict_disable_external_lag_penalty": True,
    "oracle_lag_experiment_enable": True,
    "oracle_lag_experiment_force_plot": True,
    "plot_use_lag_comp_phase": False,
}


# ==============================================================================
# ========================= 模块一：环境控制与通用工具 =========================
# ==============================================================================
def apply_env_overrides():
    """可选的环境变量覆盖开关。

    方便在不修改代码的情况下，临时切换数据目录/报告/输出目录：
    - `M9_DATA_DIR`
    - `M9_QUALITY_REPORT_PATH`
    - `M9_SAVE_DIR`
    """
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
    return "set_1.14" if "1.14" in name else "set_1"

def _trial_key(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


# ==============================================================================
# ========================= 模块二：数据清洗与质量评估 =========================
# ==============================================================================
class AdaptiveScaler:
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

def phase_from_labels(labels: np.ndarray) -> np.ndarray:
    phase = np.arctan2(labels[:, 0], labels[:, 1])
    return (phase + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

def trial_quality_score(features: np.ndarray, labels: np.ndarray):
    """在线估计一个 trial 的“可用程度/可信度”，用于训练加权或过滤。

    主要由几部分组成：
    - 饱和比例：特征落在 `[sensor_low_clip, sensor_high_clip]` 外的比例
    - 相位覆盖：相位分布的集中程度（95%~5% 分位差）
    - 有效周期：相位展开后的覆盖周期数（长度/起止差）
    - 短样本惩罚：帧数太短时对质量分打折

    最终返回：(score, sat_ratio, phase_coverage, effective_cycles)
    """
    phase = phase_from_labels(labels)
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    n_frames = int(len(labels))
    effective_cycles = float(unwrapped[-1] - unwrapped[0]) if len(unwrapped) > 1 else 0.0
    phase_coverage = float(np.quantile(phase, 0.95) - np.quantile(phase, 0.05))
    
    sat_mask = (features <= CONFIG["sensor_low_clip"]) | (features >= CONFIG["sensor_high_clip"])
    sat_ratio = float(sat_mask.mean())

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

    warn_f = int(CONFIG.get("short_clip_warn_frames", 120))
    drop_f = int(CONFIG.get("short_clip_drop_frames", 95))
    low_w = float(CONFIG.get("short_clip_penalty_weight", 0.55))
    
    if n_frames < warn_f:
        alpha = np.clip((n_frames - drop_f) / max(1, warn_f - drop_f), 0.0, 1.0)
        score *= (low_w + (1.0 - low_w) * alpha)

    score = float(np.clip(score, CONFIG["quality_floor"], CONFIG["quality_ceiling"]))
    return score, sat_ratio, phase_coverage, effective_cycles

def load_external_quality_map(path: str):
    """读取外部 quality 报告（CSV），构建 `{trial_key: quality_score}` 映射。

    若文件不存在/字段缺失/解析失败，则返回空字典，并回退到在线质量评估。
    """
    if not path or not os.path.exists(path): 
        return {}
    try: 
        df = pd.read_csv(path)
    except Exception as e: 
        return {}
        
    if not {"file", "quality_score"}.issubset(set(df.columns)): 
        return {}

    qmap = {}
    for _, row in df.iterrows():
        try:
            raw_score = float(row["quality_score"])
            clamped_score = float(np.clip(raw_score, CONFIG["quality_floor"], CONFIG["quality_ceiling"]))
            qmap[_trial_key(str(row["file"]))] = clamped_score
        except Exception: 
            continue
    return qmap

def discover_latest_lag_report(save_dir: str) -> str:
    """当你启用 lag 风险惩罚但没显式指定报告路径时，
    从 `save_dir` 同级目录里推断最近一次的 `all_folds_val_trial_mae.csv`。
    """
    if not save_dir: 
        return ""
    search_pattern = os.path.join(os.path.dirname(save_dir), "ultr_fusion_model_9_raw_v*", "all_folds_val_trial_mae.csv")
    candidates =[
        p for p in glob.glob(search_pattern) 
        if os.path.abspath(os.path.dirname(p)) != os.path.abspath(save_dir)
    ]
    if not candidates: 
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def load_lag_risk_map(path: str):
    if not path or not os.path.exists(path): 
        return {}
    try: 
        df = pd.read_csv(path)
    except Exception: 
        return {}
        
    if not {"trial_name", "lag_shift"}.issubset(set(df.columns)): 
        return {}
    
    out = {}
    for key, part in df.groupby(df["trial_name"].astype(str).map(_trial_key)):
        out[key] = float(np.mean(np.abs(part["lag_shift"].values.astype(np.float32))))
    return out


# ==============================================================================
# ========================= 模块三：核心验证指标与补偿 =========================
# ==============================================================================
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
        out[-shift:] = phase[:shift]
        out[:-shift] = phase[0]
    return out

def best_lag_compensation(pred_phase: np.ndarray, true_phase: np.ndarray, max_shift: int):
    """在 `[-max_shift, max_shift]` 范围内做“相位滞后补偿”。

    用边界保持（edge hold）方式移动 pred，相当于允许整体滞后对齐，
    选择使圆周 MAE 最小的 shift，返回：(best_phase, best_shift, best_mae)
    """
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
    
    if np.any(edge_mask):
        return circular_mae(p[edge_mask], t[edge_mask])
    return float("nan")

def estimate_cycle_frames(phase: np.ndarray) -> float:
    phase = np.asarray(phase, dtype=np.float32)
    if len(phase) < 3: 
        return float("nan")
        
    cycles = float(np.unwrap(phase * 2 * np.pi)[-1] - np.unwrap(phase * 2 * np.pi)[0]) / (2 * np.pi)
    if cycles > 1e-6:
        return float(len(phase) / cycles)
    return float("nan")

def dynamic_edge_ratio_from_phase(phase: np.ndarray) -> float:
    base_ratio = float(CONFIG.get("edge_ratio", 0.15))
    if not bool(CONFIG.get("edge_dynamic_enable", True)): 
        return base_ratio
        
    cycle_frames = estimate_cycle_frames(phase)
    if not np.isfinite(cycle_frames): 
        return base_ratio
        
    target_frames = float(CONFIG.get("edge_target_frames", 10))
    ratio = target_frames / max(1.0, cycle_frames)
    
    return float(np.clip(
        ratio, 
        CONFIG.get("edge_ratio_min", 0.08), 
        CONFIG.get("edge_ratio_max", 0.24)
    ))

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
    steps = np.abs(circular_delta(phase[1:], phase[:-1]))
    return float(np.mean(steps > threshold))

def spike_rate(phase: np.ndarray, threshold: float) -> float:
    return jump_rate(phase, threshold) 


# ==============================================================================
# ========================= 模块四：后处理平滑策略引擎 =========================
# ==============================================================================
def phase_from_preds(preds_raw: np.ndarray) -> np.ndarray:
    raw_rad = np.arctan2(preds_raw[:, 0], preds_raw[:, 1])
    return (raw_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)

def suppress_phase_spikes(phase: np.ndarray, mad_k: float, abs_step_th: float) -> np.ndarray:
    """基于 MAD 的异常步长抑制（用于去“尖刺跳变”）。

    思路：
    1) 先计算相邻帧的圆周步进 `steps`
    2) 用局部中位数估计“正常步进”，残差残差 -> 估计 MAD
    3) 若残差超过阈值且步进幅度也足够大，则把尖刺步进裁剪/替换成中位步进
    4) 通过累加步进重建整段相位（mod 1.0）
    """
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

def _smooth_robust_tracker_v2(base, preds_raw: np.ndarray) -> np.ndarray:
    tracker = base.RobustPhaseTracker(alpha=CONFIG["tracker_alpha"], beta=CONFIG["tracker_beta"])
    phases =[]
    max_step = float(CONFIG.get("tracker_max_step", 0.12))
    
    wrap_hyst = bool(CONFIG.get("wrap_hysteresis_enable", True))
    wrap_armed = False
    wrap_arm_high = float(CONFIG.get("wrap_arm_high", 0.90))
    wrap_release_low = float(CONFIG.get("wrap_release_low", 0.10))
    
    for s, c in preds_raw:
        p = tracker.update(s, c)
        if not phases:
            phases.append(p)
            continue
            
        prev = phases[-1]
        delta = p - prev
        
        if delta > 0.5: delta -= 1.0
        elif delta < -0.5: delta += 1.0

        if wrap_hyst:
            if prev >= wrap_arm_high: 
                wrap_armed = True
            
            wrap_reset = False
            if wrap_armed and (p <= wrap_release_low): 
                wrap_armed = False
                wrap_reset = True
            elif prev < 0.5 and p > 0.5: 
                wrap_armed = False
        else:
            wrap_reset = (prev > 0.85) and (p < 0.15)

        if not wrap_reset: 
            p = (prev + float(np.clip(delta, -max_step, max_step))) % 1.0
        phases.append(p)

    out = np.asarray(phases, dtype=np.float32)
    if bool(CONFIG.get("anti_spike_enable", True)):
        out = suppress_phase_spikes(
            out, 
            float(CONFIG.get("anti_spike_mad_k", 3.0)), 
            float(CONFIG.get("anti_spike_abs_step", 0.06))
        )
    return out

def smooth_raw_monotonic(preds_raw: np.ndarray, min_step: float, max_step: float) -> np.ndarray:
    """把 raw 相位转换为“近似单调”的轨迹：限制相邻步进步长范围。

    会对步进取局部中位数，再裁剪到 `[min_step, max_step]`，最后累加重建相位。
    """
    phase = phase_from_preds(preds_raw)
    if len(phase) < 3: 
        return phase.astype(np.float32)
        
    steps = circular_delta(phase[1:], phase[:-1]).astype(np.float32)
    pad = np.pad(steps, (1, 1), mode="edge")
    med = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]], axis=1), axis=1).astype(np.float32)
    clean_steps = np.clip(med, min_step, max_step)
    
    out = np.empty_like(phase, dtype=np.float32)
    out[0] = float(phase[0])
    for i in range(len(clean_steps)): 
        out[i + 1] = (out[i] + clean_steps[i]) % 1.0
    return out

def circular_mean_stack(phases, weights=None) -> np.ndarray:
    """把多个相位序列做圆周空间的加权平均。

    由于相位是周期变量，不能直接对数值平均。
    这里通过把相位映射到单位圆：
    - sin/ cos 分别加权平均
    - 再用 atan2 还原相位（仍映射到 0~1）
    """
    mats =[np.atleast_1d(np.asarray(p, dtype=np.float32)) for p in phases if p is not None]
    if not mats: 
        return np.array([], dtype=np.float32)
    if len(mats) == 1: 
        return mats[0].astype(np.float32)

    arr = np.stack(mats, axis=0)
    if arr.ndim == 1: 
        arr = arr[:, None]
        
    if weights is None:
        w2d = np.ones((arr.shape[0], arr.shape[1]), dtype=np.float32) / float(arr.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        s = float(np.sum(w))
        w = np.ones_like(w) / float(len(w)) if s <= 1e-8 else (w / s)
        w2d = np.repeat(w[:, None], arr.shape[1], axis=1)

    mean_sin = np.sum(np.sin(arr * 2 * np.pi) * w2d, axis=0)
    mean_cos = np.sum(np.cos(arr * 2 * np.pi) * w2d, axis=0)
    return ((np.arctan2(mean_sin, mean_cos) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)).astype(np.float32)

def _smooth_hybrid_ensemble(base, preds_raw: np.ndarray) -> np.ndarray:
    raw_phase = phase_from_preds(preds_raw)
    adaptive = np.asarray(smooth_predictions(base, preds_raw, strategy="adaptive_tracker"), dtype=np.float32)
    mono = np.asarray(smooth_predictions(base, preds_raw, strategy="raw_monotonic"), dtype=np.float32)
    
    smart = None
    if hasattr(base, "SmartGatedPhaseTracker"):
        smart = np.asarray(smooth_predictions(base, preds_raw, strategy="smart_tracker"), dtype=np.float32)

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
    w_smart = 0.15 + 0.15 * (1.0 - conf) if smart is not None else np.zeros_like(w_raw)
    
    w_sum = np.maximum(1e-6, w_raw + w_adp + w_mono + w_smart)

    out = np.zeros_like(raw_phase, dtype=np.float32)
    for i in range(len(raw_phase)):
        local_tracks = [raw_phase[i], adaptive[i], mono[i]]
        local_weights = [w_raw[i]/w_sum[i], w_adp[i]/w_sum[i], w_mono[i]/w_sum[i]]
        if smart is not None:
            local_tracks.append(smart[i])
            local_weights.append(w_smart[i]/w_sum[i])
        out[i] = float(circular_mean_stack(local_tracks, local_weights))

    if bool(CONFIG.get("anti_spike_enable", True)):
        mad_k = float(CONFIG.get("anti_spike_mad_k", 3.0))
        abs_step_th = float(CONFIG.get("anti_spike_abs_step", 0.06))
        out = suppress_phase_spikes(out, mad_k, abs_step_th)
    return out

def smooth_predictions(base, preds_raw, strategy=None):
    strategy = str(strategy or CONFIG.get("smoothing_strategy", "perfect_sawtooth")).lower()
    preds_raw = np.asarray(preds_raw, dtype=np.float32)

    # ================== 终极法 1：瞬时步频积分重建法 ==================
    if strategy == "perfect_sawtooth":
        raw_rad = np.arctan2(preds_raw[:, 0], preds_raw[:, 1])
        raw_phase = (raw_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        
        unwrapped = np.unwrap(raw_phase * 2 * np.pi) / (2 * np.pi)
        steps = np.diff(unwrapped)
        steps = np.insert(steps, 0, steps[0])
        steps = np.clip(steps, 0.005, 0.08)
        
        k_size = min(15, len(steps))
        win = min(15, len(steps))
        
        if k_size % 2 == 0: 
            k_size -= 1
        if k_size >= 3: 
            steps = medfilt(steps, kernel_size=k_size)
        if win >= 3: 
            steps = np.convolve(steps, np.ones(win)/win, mode='same')
            
        reconstructed = np.cumsum(steps)
        drift = unwrapped - reconstructed
        
        if len(drift) > 15:
            b, a = butter(N=2, Wn=0.015, btype="low")
            drift = filtfilt(b, a, drift)
            
        final_unwrapped = reconstructed + drift + 0.015
        return (final_unwrapped % 1.0).tolist()

    # ================== 终极法 2：锚点对齐线性插值法 ==================
    if strategy == "anchor_sync":
        raw_rad = np.arctan2(preds_raw[:, 0], preds_raw[:, 1])
        raw_phase = (raw_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        raw_unwrapped = np.unwrap(raw_phase * 2 * np.pi) / (2 * np.pi)
        
        if len(raw_unwrapped) > 15:
            b, a = butter(N=2, Wn=0.1, btype="low")
            smooth_phase = filtfilt(b, a, raw_unwrapped) % 1.0
        else:
            smooth_phase = raw_unwrapped % 1.0
            
        diffs = np.diff(smooth_phase)
        anchors = np.where(diffs < -0.5)[0] + 1
        
        if len(anchors) == 0: 
            return smooth_raw_monotonic(preds_raw, 0.005, 0.08).tolist()
            
        out_phase = np.zeros_like(raw_phase)
        for i in range(len(anchors) - 1):
            start = anchors[i]
            end = anchors[i+1]
            out_phase[start:end] = np.linspace(0, 1, end - start, endpoint=False)
            
        if len(anchors) > 1:
            avg_cycle = np.mean(np.diff(anchors))
        else:
            total_cycles = max(0.5, (raw_unwrapped[-1] - raw_unwrapped[0]))
            avg_cycle = len(raw_phase) / total_cycles
            
        first_anchor = anchors[0]
        last_anchor = anchors[-1]
        
        if first_anchor > 0: 
            out_phase[:first_anchor] = (1.0 - np.arange(first_anchor, 0, -1) / avg_cycle) % 1.0
        if last_anchor < len(raw_phase): 
            out_phase[last_anchor:] = (np.arange(0, len(raw_phase) - last_anchor) / avg_cycle) % 1.0
            
        return out_phase.tolist()

    # ---------------- 传统兼容策略 ----------------
    if strategy == "raw": 
        return phase_from_preds(preds_raw).tolist()

    if strategy == "robust_tracker": 
        tracker = base.RobustPhaseTracker(
            alpha=CONFIG["tracker_alpha"], 
            beta=CONFIG["tracker_beta"]
        )
        return[tracker.update(s, c) for s, c in preds_raw]

    if strategy == "robust_tracker_v2": 
        return _smooth_robust_tracker_v2(base, preds_raw).tolist()

    if strategy == "raw_monotonic":
        min_step = float(CONFIG.get("mono_min_step", -0.005))
        max_step = float(CONFIG.get("mono_max_step", 0.080))
        out = smooth_raw_monotonic(preds_raw, min_step, max_step)
        
        if bool(CONFIG.get("anti_spike_enable", True)):
            mad_k = float(CONFIG.get("anti_spike_mad_k", 3.0))
            abs_step_th = float(CONFIG.get("anti_spike_abs_step", 0.06))
            out = suppress_phase_spikes(out, mad_k, abs_step_th)
        return out.tolist()

    if strategy == "lowpass":
        smoother = base.EnhancedVectorSmoother(alpha=CONFIG["ema_alpha"])
        base_phases = [smoother.update(s, c) for s, c in preds_raw]
        unwrapped_phase = np.unwrap(np.array(base_phases) * 2 * np.pi)
        
        b, a = butter(N=CONFIG["butter_order"], Wn=CONFIG["butter_cutoff"], btype="low")
        smoothed = filtfilt(b, a, unwrapped_phase)
        return ((smoothed / (2 * np.pi)) % 1.0).tolist()

    if strategy == "smart_tracker" and hasattr(base, "SmartGatedPhaseTracker"): 
        tracker = base.SmartGatedPhaseTracker()
        return[tracker.update(s, c) for s, c in preds_raw]

    if strategy == "hybrid_ensemble": 
        return _smooth_hybrid_ensemble(base, preds_raw).tolist()

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
            mad_k = float(CONFIG.get("anti_spike_mad_k", 3.0))
            abs_step_th = float(CONFIG.get("anti_spike_abs_step", 0.06))
            out = suppress_phase_spikes(out, mad_k, abs_step_th)
        return out.tolist()

    # 兜底默认策略
    smoother = base.EnhancedVectorSmoother(alpha=CONFIG["ema_alpha"])
    return [smoother.update(s, c) for s, c in preds_raw]


def resolve_strategy_candidates(base):
    """把 `CONFIG['strategy_candidates']` 解析为最终可用策略列表，并按约束做过滤。

    - 自动处理大小写/别名：`basic` -> `robust_tracker`
    - `smart_tracker` 需要 base 中存在对应追踪器类
    - 若开启 `strict_disable_i1_usage_enable`：只保留“更保守”的策略集合
    """
    known = {
        "raw", "robust_tracker", "robust_tracker_v2", "adaptive_tracker", 
        "raw_monotonic", "lowpass", "smart_tracker", "hybrid_ensemble", 
        "basic", "perfect_sawtooth", "anchor_sync"
    }
    raw_candidates = CONFIG.get("strategy_candidates", [CONFIG.get("smoothing_strategy", "adaptive_tracker")])
    if isinstance(raw_candidates, str): 
        raw_candidates = [raw_candidates]

    if bool(CONFIG.get("strict_disable_i1_usage_enable", False)):
        allowed = {"raw", "robust_tracker", "robust_tracker_v2", "adaptive_tracker", "smart_tracker"}
        raw_candidates =[str(s).lower().strip() for s in raw_candidates if str(s).lower().strip() in allowed]

    out =[]
    for key in (str(s).lower().strip() for s in raw_candidates if str(s).lower().strip()):
        if key not in known: 
            print(f"[StrategyEval] 跳过未知策略: {key}")
            continue
        if key == "smart_tracker" and not hasattr(base, "SmartGatedPhaseTracker"): 
            print("[StrategyEval] 跳过 smart_tracker（base 无）。")
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
    
    pred_oracle = pred_phases
    lag_shift = 0
    mae_oracle = mae_no_lag
    
    if bool(CONFIG.get("lag_comp_enable", True)):
        max_shift = int(CONFIG.get("lag_comp_max_shift", 6))
        pred_oracle, lag_shift, mae_oracle = best_lag_compensation(pred_phases, true_phases, max_shift)

    use_no_lag = strict_eval and primary_mode == "no_lag"
    pred_eval = pred_phases if use_no_lag else pred_oracle
    trial_mae = mae_no_lag if use_no_lag else mae_oracle
    
    jump_th = float(CONFIG.get("diag_jump_threshold", 0.08))
    spike_th = float(CONFIG.get("diag_spike_threshold", 0.06))
    jr = jump_rate(pred_phases, jump_th)
    sr = spike_rate(pred_phases, spike_th)
    
    e_mae = edge_phase_mae(pred_eval, true_phases, dynamic_edge_ratio_from_phase(true_phases))
    s_mae = slope_phase_mae(pred_eval, true_phases)
    
    e_mae_safe = float(trial_mae if not np.isfinite(e_mae) else e_mae)
    s_mae_safe = float(0.0 if not np.isfinite(s_mae) else s_mae)
    
    edge_w = float(CONFIG.get("strategy_score_edge_weight", 0.15))
    jump_w = float(CONFIG.get("strategy_score_jump_weight", 0.05))
    spike_w = float(CONFIG.get("strategy_score_spike_weight", 0.05))
    slope_w = float(CONFIG.get("strategy_score_slope_weight", 0.08))
    
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
        "composite_score": composite
    }

def auto_calibrate_postprocess(scored_trials, tag: str = "global"):
    if not bool(CONFIG.get("auto_calibrate_postprocess", True)): 
        return
        
    step_abs_list, cycle_frames = [],[]
    for _, _, label, *_rest in scored_trials:
        phase = phase_from_labels(label)
        if len(phase) > 1: 
            step_abs_list.append(np.abs(circular_delta(phase[1:], phase[:-1])).astype(np.float32))
            
        cyc = _rest[-1]
        if cyc and cyc > 1e-6: 
            cycle_frames.append(float(len(label)) / float(cyc))
            
    if not step_abs_list: 
        return

    step_abs_concat = np.concatenate(step_abs_list)
    q50, q90, q95 = np.quantile(step_abs_concat, [0.50, 0.90, 0.95])
    
    raw_mono_max = float(np.clip(q95 * 1.35, 0.05, 0.14))
    raw_mono_min_abs = float(np.clip(q50 * 0.35, 0.002, 0.02))
    raw_jump_th = float(np.clip(q90 * 1.15, 0.06, 0.12))
    raw_spike_th = float(np.clip(q95 * 0.90, 0.05, 0.11))
    
    median_cycle = np.median(cycle_frames) if cycle_frames else (6.0 / 0.22)
    raw_lag_max = int(np.clip(round(median_cycle * 0.22), 3, 10))

    mode = str(CONFIG.get("auto_calibrate_mode", "guarded")).lower()
    if mode != "legacy":
        min_trials = int(CONFIG.get("auto_calibrate_min_trials", 24))
        if len(scored_trials) < min_trials:
            return
            
        if len(cycle_frames) >= 4:
            cv = float(np.std(cycle_frames) / max(1e-6, np.mean(cycle_frames)))
            max_cv = float(CONFIG.get("auto_calibrate_cycle_cv_max", 0.22))
            if cv > max_cv:
                return

    def _b_rel(anc, prp, lo, hi):
        max_change = float(CONFIG.get("auto_calibrate_max_rel_change", 0.20))
        lower_bound = anc * (1.0 - max_change)
        upper_bound = anc * (1.0 + max_change)
        return float(np.clip(prp, max(lo, lower_bound), min(hi, upper_bound)))

    if mode == "legacy":
        mono_max = raw_mono_max
        mono_min = -raw_mono_min_abs
        jump_th = raw_jump_th
        spike_th = raw_spike_th
        lag_max = raw_lag_max
    else:
        mono_max = _b_rel(float(CONFIG.get("mono_max_step", 0.08)), raw_mono_max, 0.05, 0.14)
        mono_min = -_b_rel(abs(float(CONFIG.get("mono_min_step", -0.005))), raw_mono_min_abs, 0.002, 0.02)
        jump_th = _b_rel(float(CONFIG.get("diag_jump_threshold", 0.08)), raw_jump_th, 0.06, 0.12)
        
        base_spike = float(CONFIG.get("diag_spike_threshold", 0.06))
        spike_th = _b_rel(base_spike, raw_spike_th, 0.05, 0.11)
        spike_th = min(spike_th, jump_th)
        
        base_lag = float(CONFIG.get("lag_comp_max_shift", 6))
        lag_max = int(np.clip(round(_b_rel(base_lag, float(raw_lag_max), 4, 8)), 4, 8))

    CONFIG.update({
        "mono_max_step": mono_max, 
        "mono_min_step": mono_min, 
        "diag_jump_threshold": jump_th, 
        "diag_spike_threshold": spike_th, 
        "lag_comp_max_shift": lag_max
    })
    
    try:
        filename = f"auto_calibration_snapshot_{tag}.json" if tag else "auto_calibration_snapshot.json"
        with open(os.path.join(CONFIG["save_dir"], filename), "w", encoding="utf-8") as f:
            data = {
                "mode": mode, 
                "n_trials": int(len(scored_trials)), 
                "applied": {
                    "mono_max_step": float(mono_max), 
                    "mono_min_step": float(mono_min), 
                    "jump_threshold": float(jump_th), 
                    "spike_threshold": float(spike_th), 
                    "lag_comp_max_shift": int(lag_max)
                }
            }
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception: 
        pass


# ==============================================================================
# ========================= 模块五：交叉验证与主管道 ===========================
# ==============================================================================
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
        rng.shuffle(ranked_groups)
        ranked_groups = sorted(ranked_groups, key=lambda g: (-group_counts[g], str(g)))

    fold_groups = [set() for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    
    for g in ranked_groups:
        candidates = [i for i, sz in enumerate(fold_sizes) if sz == min(fold_sizes)]
        fidx = int(rng.choice(candidates)) if len(candidates) > 1 else candidates[0]
        fold_groups[fidx].add(g)
        fold_sizes[fidx] += int(group_counts[g])

    splits =[]
    for fidx in range(n_splits):
        val_mask = np.array([g in fold_groups[fidx] for g in groups], dtype=bool)
        train_idx = idx_all[~val_mask]
        val_idx = idx_all[val_mask]
        if len(val_idx) > 0 and len(train_idx) > 0: 
            splits.append((train_idx, val_idx))
            
    return splits, fold_groups, fold_sizes

def main():
    """主程序：完成 k-fold 训练 + 后处理策略评估/选择。

    关键流程：
    1) apply_env_overrides：可选环境变量覆盖
    2) 加载 base 脚本并拿到所有 trial
    3) 计算在线 quality 并可合并外部 quality 报告
    4) 可选启用 oracle lag/滞后风险惩罚（影响评估指标）
    5) 对策略候选池进行折内评估与折外汇总（并输出 report/csv/图）
    """
    apply_env_overrides()
    
    if bool(CONFIG.get("oracle_lag_experiment_enable", False)):
        CONFIG["strict_eval_primary"] = "oracle_lag"
        if bool(CONFIG.get("oracle_lag_experiment_force_plot", True)): 
            CONFIG["plot_use_lag_comp_phase"] = True
        
    np.random.seed(CONFIG.get("random_seed", 42))
    tf.random.set_seed(CONFIG.get("random_seed", 42))
    
    base = load_base_module(CONFIG["base_train_script"])
    if not os.path.exists(CONFIG["save_dir"]): 
        os.makedirs(CONFIG["save_dir"])

    print("1. 加载并评分所有 trial ...")
    raw_trials = base.load_all_data_without_split(CONFIG["data_dir"])
    if not raw_trials: 
        raise ValueError("未找到可训练数据。")

    external_qmap = load_external_quality_map(CONFIG.get("quality_report_path", ""))
    scored_trials =[]
    dropped_short_rows =[]
    
    report_headers = "name,subject,domain,n_samples,quality,quality_source,saturation_ratio,phase_coverage,effective_cycles"
    report_lines = [report_headers]
    
    # ------------------ 1. 数据质量过滤与打分 ------------------
    for name, feat, label in raw_trials:
        drop_enable = CONFIG.get("drop_short_trials_enable", False)
        drop_frames = int(CONFIG.get("drop_short_trials_frames", 95))
        
        if drop_enable and len(label) < drop_frames:
            dropped_short_rows.append({
                "name": name, 
                "n_samples": int(len(label)), 
                "drop_rule": f"short_clip<{drop_frames}"
            })
            continue

        subject = parse_subject_id(name)
        domain = parse_domain(name)
        q_online, sat, cov, cyc = trial_quality_score(feat, label)
        
        external_score = external_qmap.get(_trial_key(name))
        if external_score is None:
            q, q_source = q_online, "online"
        else:
            q, q_source = external_score, "external_report"
            
        scored_trials.append((name, feat, label, subject, domain, q, sat, cov, cyc))
        report_lines.append(f"{name},{subject},{domain},{len(label)},{q:.4f},{q_source},{sat:.4f},{cov:.4f},{cyc:.4f}")

    if dropped_short_rows:
        drop_df = pd.DataFrame(dropped_short_rows)
        drop_df.to_csv(os.path.join(CONFIG["save_dir"], "dropped_short_trials.csv"), index=False, encoding="utf-8-sig")
        
    with open(os.path.join(CONFIG["save_dir"], "trial_quality_report.csv"), "w", encoding="utf-8") as f: 
        f.write("\n".join(report_lines))

    # ------------------ 2. 滞后风险惩罚配置 ------------------
    lag_risk_path = str(CONFIG.get("lag_risk_report_path", "")).strip()
    strict_eval = bool(CONFIG.get("strict_eval_enable", True))
    disable_ext_lag = bool(CONFIG.get("strict_disable_external_lag_penalty", True))
    
    lag_penalty_flag = False if strict_eval and disable_ext_lag else bool(CONFIG.get("lag_penalty_enable", True))
    
    if not lag_risk_path and lag_penalty_flag:
        lag_risk_path = discover_latest_lag_report(CONFIG["save_dir"])
        
    lag_risk_map = load_lag_risk_map(lag_risk_path) if lag_penalty_flag else {}

    # ------------------ 3. 交叉验证切分 ------------------
    groups = [t[3] for t in scored_trials]
    unique_group_count = len(set(groups))
    target_folds = 3 if len(scored_trials) < 15 and CONFIG["k_folds"] > 3 else CONFIG["k_folds"]
    n_splits = min(target_folds, unique_group_count)

    cv_strategy = str(CONFIG.get("cv_strategy", "balanced_group")).lower()
    if cv_strategy == "groupkfold":
        fold_splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(scored_trials)), groups=groups))
    else:
        fold_splits, _, _ = build_balanced_group_splits(
            groups=groups, 
            n_splits=n_splits, 
            random_seed=int(CONFIG.get("cv_random_seed", 42)), 
            shuffle_groups=bool(CONFIG.get("cv_shuffle_groups", True))
        )

    strategy_candidates = resolve_strategy_candidates(base)
    calib_keys =["mono_max_step", "mono_min_step", "diag_jump_threshold", "diag_spike_threshold", "lag_comp_max_shift"]
    calib_base = {k: CONFIG.get(k) for k in calib_keys}
    
    all_fold_maes =[]
    all_fold_trial_details =[]

    # ==========================================================
    # =================== 开始折叠循环 =========================
    # ==========================================================
    for fold, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n================ 开始训练第 {fold}/{len(fold_splits)} 折 ================")
        
        train_trials =[scored_trials[i] for i in train_idx]
        val_trials = [scored_trials[i] for i in val_idx]
        
        # 恢复参数并进行自校准
        for k in calib_base: 
            CONFIG[k] = calib_base[k]
        auto_calibrate_postprocess(train_trials, tag=f"fold_{fold}")

        # ---------------- 4. 训练集样本权重构建 ----------------
        domain_counter = Counter([t[4] for t in train_trials])
        domain_weight_map = {}
        for d, c in domain_counter.items():
            w = len(train_trials) / max(1.0, len(domain_counter) * c) if CONFIG["domain_balance_enable"] else 1.0
            if bool(CONFIG.get("manual_weight_overrides_enable", False)):
                w *= float(CONFIG.get("domain_weight_overrides", {}).get(d, 1.0))
            domain_weight_map[d] = float(np.clip(w, CONFIG.get("domain_weight_min", 0.75), CONFIG.get("domain_weight_max", 1.20)))

        aug_train_features, aug_train_labels = [],[]
        all_train_feat_concat, train_meta = [],[]
        
        lag_abs_shift = float(CONFIG.get("lag_penalty_abs_shift", 4))
        risky_train_keys = {_trial_key(t[0]) for t in train_trials if lag_risk_map.get(_trial_key(t[0]), 0) >= lag_abs_shift} if lag_risk_map else set()
        
        lag_max_ratio = float(CONFIG.get("lag_penalty_max_ratio", 0.35))
        apply_lag_penalty = bool(lag_risk_map) and (len(risky_train_keys) / max(1, len(train_trials)) <= lag_max_ratio)

        for trial_name, feat, label, subject, domain, quality, *_ in train_trials:
            subject_w = 1.0
            if bool(CONFIG.get("manual_weight_overrides_enable", False)):
                sub_val = CONFIG["subject_weight_map"].get(str(subject), CONFIG["subject_weight_default"])
                subject_w = float(np.clip(sub_val, CONFIG["subject_weight_min"], CONFIG["subject_weight_max"]))
                
            lag_w = 1.0
            if apply_lag_penalty and _trial_key(trial_name) in risky_train_keys:
                lag_w = float(CONFIG.get("lag_penalty_weight", 0.85))
                
            combined_w = quality * domain_weight_map.get(domain, 1.0) * subject_w * lag_w
            trial_w = float(np.clip(combined_w, CONFIG.get("sample_weight_min", 0.70), CONFIG.get("sample_weight_max", 1.30)))
            
            for factor in CONFIG["augment_factors"]:
                f_aug, l_aug = base.augment_time_warp(feat, label, factor)
                aug_train_features.append(f_aug)
                aug_train_labels.append(l_aug)
                all_train_feat_concat.append(f_aug)
                train_meta.append(trial_w)

        # ---------------- 5. 特征缩放与降维 ----------------
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

        # ---------------- 6. 构造训练与验证序列 ----------------
        X_train_seqs, y_train_seqs, train_weights = [], [], []
        seq_len = CONFIG["sequence_length"]
        
        for feat, label, trial_w in zip(aug_train_features, aug_train_labels, train_meta):
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = base.create_sequences(f_pca, label, seq_len)
            
            if len(X_seq) == 0: 
                continue
                
            X_train_seqs.append(X_seq)
            y_train_seqs.append(y_seq)

            if CONFIG["edge_weight_enable"]:
                true_phases = (np.arctan2(y_seq[:, 0], y_seq[:, 1]) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
                edge_ratio_local = dynamic_edge_ratio_from_phase(true_phases)
                w_seq = np.ones(len(y_seq), dtype=np.float32)
                
                edge_mask = (true_phases < edge_ratio_local) | (true_phases > (1.0 - edge_ratio_local))
                w_seq[edge_mask] = CONFIG["edge_weight"]
            else:
                w_seq = np.ones(len(X_seq), dtype=np.float32)
                
            train_weights.append((w_seq * trial_w).astype(np.float32))

        X_train = np.vstack(X_train_seqs)
        y_train = np.vstack(y_train_seqs)
        sample_weights = np.concatenate(train_weights).astype(np.float32)
        
        idx = np.random.permutation(len(X_train))
        X_train, y_train, sample_weights = X_train[idx], y_train[idx], sample_weights[idx]

        X_val_seqs, y_val_seqs = [],[]
        for _, feat, label, *_ in val_trials:
            f_scaled = scaler.transform(feat)
            f_pca = pca.transform(f_scaled)
            X_seq, y_seq = base.create_sequences(f_pca, label, seq_len)
            if len(X_seq) > 0: 
                X_val_seqs.append(X_seq)
                y_val_seqs.append(y_seq)
                
        if not X_val_seqs: 
            continue
        X_val = np.vstack(X_val_seqs)
        y_val = np.vstack(y_val_seqs)

        # ---------------- 7. 训练模型 ----------------
        base.CONFIG.update({"learning_rate": CONFIG["learning_rate"], "l2_reg": base.CONFIG.get("l2_reg", 0.001)})
        model = base.build_fusion_model(input_shape=(seq_len, CONFIG["pca_components"]))

        callbacks = [
            ModelCheckpoint(os.path.join(CONFIG["save_dir"], f"best_model_fold_{fold}.keras"), monitor="val_phase_mae", save_best_only=True, mode="min"),
            ReduceLROnPlateau(monitor="val_phase_mae", factor=CONFIG["reduce_lr_factor"], patience=CONFIG["reduce_lr_patience"], verbose=0, mode="min"),
            EarlyStopping(monitor="val_phase_mae", patience=CONFIG["early_stop_patience"], restore_best_weights=True, verbose=0, mode="min"),
        ]

        model.fit(
            X_train, y_train, 
            sample_weight=sample_weights, 
            validation_data=(X_val, y_val), 
            epochs=CONFIG["epochs"], 
            batch_size=CONFIG["batch_size"], 
            verbose=2, 
            callbacks=callbacks
        )

        # ---------------- 8. 模型验证与多策略评估 ----------------
        print(f"\n正在评估第 {fold} 折验证集...")
        val_cache =[]
        
        for trial_name, test_feat, test_label, subject, domain, quality, sat, cov, cyc in val_trials:
            f_scaled = scaler.transform(test_feat)
            f_pca = pca.transform(f_scaled)
            X_test_trial, y_test_trial = base.create_sequences(f_pca, test_label, seq_len)
            
            if len(X_test_trial) == 0: 
                continue
                
            preds_raw = model.predict(X_test_trial, verbose=0)
            preds_raw = np.asarray(preds_raw, dtype=np.float32)
            true_rad = np.arctan2(y_test_trial[:, 0], y_test_trial[:, 1])
            true_phases = np.asarray((true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi), dtype=np.float32)
            
            val_cache.append({
                "trial_name": trial_name, 
                "subject": subject, 
                "domain": domain, 
                "quality": float(quality), 
                "sat": float(sat), 
                "cov": float(cov), 
                "cyc": float(cyc), 
                "n_raw_frames": int(len(test_label)), 
                "n_eval_sequences": int(len(X_test_trial)), 
                "preds_raw": preds_raw, 
                "true_phases": true_phases
            })

        if not val_cache: 
            continue

        active_strats = strategy_candidates if bool(CONFIG.get("strategy_eval_enable", True)) else[str(CONFIG.get("smoothing_strategy", "adaptive_tracker")).lower()]
        strategy_rows =[]
        
        for strategy in active_strats:
            trial_scores =[]
            trial_mae_no_lag = []
            trial_mae_primary = []
            trial_edge =[]
            trial_jump = []
            trial_spike = []
            trial_slope =[]
            
            for item in val_cache:
                metrics = evaluate_one_strategy(
                    base=base, 
                    preds_raw=item["preds_raw"], 
                    true_phases=item["true_phases"], 
                    strategy=strategy, 
                    strict_eval=bool(CONFIG.get("strict_eval_enable", True)), 
                    primary_mode=str(CONFIG.get("strict_eval_primary", "no_lag")).lower()
                )
                trial_scores.append(metrics["composite_score"])
                trial_mae_no_lag.append(metrics["trial_mae_no_lag"])
                trial_mae_primary.append(metrics["trial_mae"])
                trial_edge.append(metrics["edge_mae"])
                trial_jump.append(metrics["jump_rate"])
                trial_spike.append(metrics["spike_ratio"])
                trial_slope.append(metrics["slope_mae"])

            mae_arr = np.asarray(trial_mae_primary, dtype=np.float32)
            score_mode = str(CONFIG.get("strategy_fold_score_mode", "median_mae_std")).lower()
            
            if score_mode == "mean_composite":
                fold_score = float(np.mean(trial_scores))
            else:
                std_w = float(CONFIG.get("strategy_fold_score_std_weight", 0.5))
                fold_score = float(np.nanmedian(mae_arr) + std_w * np.nanstd(mae_arr))
                
            strategy_rows.append({
                "fold": fold, 
                "strategy": strategy, 
                "fold_score_mode": score_mode, 
                "fold_score": fold_score, 
                "mean_composite_score": float(np.mean(trial_scores)), 
                "mean_trial_mae_primary": float(np.mean(trial_mae_primary)), 
                "median_trial_mae_primary": float(np.nanmedian(mae_arr)), 
                "std_trial_mae_primary": float(np.nanstd(mae_arr)), 
                "mean_trial_mae_no_lag": float(np.mean(trial_mae_no_lag)), 
                "mean_edge_mae": float(np.nanmean(np.asarray(trial_edge, dtype=np.float32))), 
                "mean_jump_rate": float(np.mean(trial_jump)), 
                "mean_spike_ratio": float(np.mean(trial_spike)), 
                "mean_slope_mae": float(np.nanmean(np.asarray(trial_slope, dtype=np.float32))), 
                "n_trials": int(len(trial_scores))
            })

        # ---------------- 9. 保存策略榜单并选取最佳策略 ----------------
        strategy_df = pd.DataFrame(strategy_rows).sort_values("fold_score", ascending=True)
        strategy_df.to_csv(os.path.join(CONFIG["save_dir"], f"fold_{fold}_strategy_benchmark.csv"), index=False, encoding="utf-8-sig")
        best_strategy = str(strategy_df.iloc[0]["strategy"])
        
        print(f"[StrategyEval] fold {fold} 最优策略: {best_strategy} | fold_score={float(strategy_df.iloc[0]['fold_score']):.5f}, MAE_primary={float(strategy_df.iloc[0]['median_trial_mae_primary']) * 100:.2f}%")

        fold_trial_maes =[]
        fold_trial_details =[]
        fold_trial_plot_cache = {}
        
        for item in val_cache:
            metrics = evaluate_one_strategy(
                base=base, 
                preds_raw=item["preds_raw"], 
                true_phases=item["true_phases"], 
                strategy=best_strategy, 
                strict_eval=bool(CONFIG.get("strict_eval_enable", True)), 
                primary_mode=str(CONFIG.get("strict_eval_primary", "no_lag")).lower()
            )
            fold_trial_maes.append(metrics["trial_mae"])
            
            fold_trial_details.append({
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
                "trial_mae": float(metrics["trial_mae"]), 
                "trial_mae_before_lag": float(metrics["trial_mae_no_lag"]), 
                "trial_mae_no_lag": float(metrics["trial_mae_no_lag"]), 
                "trial_mae_oracle_lag": float(metrics["trial_mae_oracle_lag"]), 
                "lag_shift": int(metrics["lag_shift"]), 
                "jump_rate": float(metrics["jump_rate"]), 
                "spike_ratio": float(metrics["spike_ratio"]), 
                "edge_mae": float(metrics["edge_mae"]), 
                "slope_mae": float(metrics["slope_mae"])
            })
            
            plot_phase = metrics["pred_oracle"] if bool(CONFIG.get("plot_use_lag_comp_phase", False)) else metrics["pred_phases"]
            fold_trial_plot_cache[item["trial_name"]] = {
                "true_phases": item["true_phases"].copy(), 
                "plot_phase": np.asarray(plot_phase, dtype=np.float32).copy(), 
                "trial_mae": float(metrics["trial_mae"]), 
                "strategy": best_strategy
            }
        
        current_fold_mae = float(np.mean(fold_trial_maes))
        all_fold_maes.append(current_fold_mae)
        
        metric_tag = 'no_lag' if (bool(CONFIG.get('strict_eval_enable', True)) and str(CONFIG.get('strict_eval_primary', 'no_lag')).lower() == 'no_lag') else 'oracle_lag'
        print(f"第 {fold} 折验证集平均 MAE[{metric_tag}]: {current_fold_mae * 100:.2f}% (strategy={best_strategy})")

        # ---------------- 10. 保存折内报告与绘图 ----------------
        fold_df = pd.DataFrame(fold_trial_details).sort_values("trial_mae", ascending=False)
        all_fold_trial_details.extend(fold_trial_details)
        
        fold_df.to_csv(os.path.join(CONFIG["save_dir"], f"fold_{fold}_val_trial_mae.csv"), index=False, encoding="utf-8-sig")
        fold_df.head(10).to_csv(os.path.join(CONFIG["save_dir"], f"fold_{fold}_bad_trials_top10.csv"), index=False, encoding="utf-8-sig")

        ranked_trials = fold_df["trial_name"].tolist()
        plotted =[]
        used_trial = set()
        
        rep_index_map = {
            "worst": 0, 
            "median": len(ranked_trials) // 2, 
            "best": len(ranked_trials) - 1
        }
        
        for tag, ridx in rep_index_map.items():
            if not ranked_trials or ranked_trials[ridx] in used_trial or fold_trial_plot_cache.get(ranked_trials[ridx]) is None: 
                continue
                
            trial_name = ranked_trials[ridx]
            cache = fold_trial_plot_cache.get(trial_name)
            
            used_trial.add(trial_name)
            plotted.append((tag, trial_name, cache["trial_mae"]))
            
            limit = min(500, len(cache["true_phases"]))
            
            plt.figure(figsize=(15, 6))
            plt.plot(cache["true_phases"][:limit], "k-", alpha=0.5, linewidth=3, label="Ground Truth Phase")
            plt.plot(cache["plot_phase"][:limit], "r--", linewidth=2, label=f"Predicted Phase ({cache.get('strategy', best_strategy)})")
            plt.title(f"Fold {fold} {tag.capitalize()} Trial - MAE: {cache['trial_mae'] * 100:.2f}%")
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(CONFIG["save_dir"], f"fold_{fold}_{tag}_trial.png"))
            if tag == "median": 
                plt.savefig(os.path.join(CONFIG["save_dir"], f"fold_{fold}_result.png"))
            plt.close()

        tf.keras.backend.clear_session()
        
        # 保存最佳折模型
        best_fold_idx = int(np.argmin(all_fold_maes)) + 1
        shutil.copy(os.path.join(CONFIG["save_dir"], f"scaler_fold_{best_fold_idx}.pkl"), os.path.join(CONFIG["save_dir"], "best_scaler_overall.pkl"))
        shutil.copy(os.path.join(CONFIG["save_dir"], f"pca_fold_{best_fold_idx}.pkl"), os.path.join(CONFIG["save_dir"], "best_pca_overall.pkl"))
        shutil.copy(os.path.join(CONFIG["save_dir"], f"best_model_fold_{best_fold_idx}.keras"), os.path.join(CONFIG["save_dir"], "best_model_overall.keras"))

    # ==========================================================
    # =================== 训练结束，生成全局报告 ===================
    # ==========================================================
    if all_fold_trial_details:
        all_trial_df = pd.DataFrame(all_fold_trial_details)
        all_trial_df.sort_values(["fold", "trial_mae"], ascending=[True, False]).to_csv(os.path.join(CONFIG["save_dir"], "all_folds_val_trial_mae.csv"), index=False, encoding="utf-8-sig")
        all_trial_df.sort_values("trial_mae", ascending=False).head(30).to_csv(os.path.join(CONFIG["save_dir"], "all_folds_worst_trials_top30.csv"), index=False, encoding="utf-8-sig")
        
        subject_summary = all_trial_df.groupby("subject").agg(
            n_trials=("trial_mae", "size"), 
            mean_trial_mae=("trial_mae", "mean"), 
            mean_trial_mae_before_lag=("trial_mae_before_lag", "mean"), 
            mean_abs_lag=("lag_shift", lambda s: float(np.mean(np.abs(s)))), 
            mean_edge_mae=("edge_mae", "mean"), 
            mean_jump_rate=("jump_rate", "mean"), 
            mean_spike_ratio=("spike_ratio", "mean")
        ).reset_index().sort_values("mean_trial_mae", ascending=False)
        
        subject_summary.to_csv(os.path.join(CONFIG["save_dir"], "all_folds_subject_risk_summary.csv"), index=False, encoding="utf-8-sig")

    strategy_files = sorted(glob.glob(os.path.join(CONFIG["save_dir"], "fold_*_strategy_benchmark.csv")))
    if strategy_files:
        all_rows =[pd.read_csv(p) for p in strategy_files if os.path.getsize(p) > 0]
        if all_rows:
            merged_df = pd.concat(all_rows, ignore_index=True)
            global_rank = merged_df.groupby("strategy").agg(
                n_folds=("fold", "nunique"), 
                mean_fold_score=("fold_score", "mean"), 
                mean_trial_mae_primary=("mean_trial_mae_primary", "mean"), 
                mean_median_trial_mae_primary=("median_trial_mae_primary", "mean"), 
                mean_trial_mae_no_lag=("mean_trial_mae_no_lag", "mean"), 
                mean_edge_mae=("mean_edge_mae", "mean"), 
                mean_jump_rate=("mean_jump_rate", "mean"), 
                mean_spike_ratio=("mean_spike_ratio", "mean"), 
                mean_slope_mae=("mean_slope_mae", "mean")
            ).reset_index()
            
            global_rank = global_rank.sort_values("mean_fold_score", ascending=True)
            global_rank.to_csv(os.path.join(CONFIG["save_dir"], "all_folds_strategy_benchmark.csv"), index=False, encoding="utf-8-sig")
            
            print(f"[StrategyEval] 全折最优策略: {global_rank.iloc[0]['strategy']} | "
                  f"fold_score={float(global_rank.iloc[0]['mean_fold_score']):.5f}, "
                  f"MAE_primary(mean)={float(global_rank.iloc[0]['mean_trial_mae_primary']) * 100:.2f}%")

    print(f"================================================================")
    print(f"Model 9 全部完成，平均 MAE: {np.mean(all_fold_maes) * 100:.2f}%")
    print(f"================================================================")

if __name__ == "__main__":
    main()