import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


CONFIG = {
    # 默认接新版“超声数据处理.py”的输出目录，构建 clean_v3
    "input_dir": r"D:\nightmare\Documents\SRTP\AllData\2026_1_26\Processed_Final_Training_Set_1_26_1",
    "output_dir": r"D:\nightmare\Documents\SRTP\AllData\Processed_Final_Training_Set_clean_v3",
    "report_dir": r"D:\nightmare\Documents\SRTP\LSTM\training_logs\clean_set_build_v3",
    # filter thresholds
    "sensor_low_clip": 55.0,
    "sensor_high_clip": 205.0,
    "drop_saturation_ratio": 0.22,
    "warn_saturation_ratio": 0.08,
    "min_phase_levels_hard": 12,
    "min_phase_levels_warn": 24,
    "min_phase_coverage_hard": 0.70,
    "min_phase_coverage_warn": 0.80,
    "min_effective_cycles_hard": 2.5,
    "min_effective_cycles_warn": 4.0,
    "max_cross_sheet_phase_mae": 0.03,
}


def wrapped_phase_from_sincos(sin_vals: np.ndarray, cos_vals: np.ndarray) -> np.ndarray:
    return (np.arctan2(sin_vals, cos_vals) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)


def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def phase_metrics(phase: np.ndarray) -> Dict[str, float]:
    if len(phase) < 2:
        return {
            "phase_levels": float(len(np.unique(np.round(phase, 6)))),
            "phase_coverage": 0.0,
            "effective_cycles": 0.0,
        }
    unique_vals = np.unique(np.round(phase, 6))
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    return {
        "phase_levels": float(len(unique_vals)),
        "phase_coverage": float(np.quantile(phase, 0.95) - np.quantile(phase, 0.05)),
        "effective_cycles": float(unwrapped[-1] - unwrapped[0]),
    }


def reconstruct_phase(sin_vals: np.ndarray, cos_vals: np.ndarray) -> np.ndarray:
    phase = wrapped_phase_from_sincos(sin_vals, cos_vals)
    if len(phase) < 7:
        return phase
    raw_u = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    mono_u = np.maximum.accumulate(raw_u)

    # Window scales with sequence length; enforce odd.
    w = min(31, len(mono_u) - (1 - len(mono_u) % 2))
    if w < 7:
        w = 7 if len(mono_u) >= 7 else len(mono_u) - (1 - len(mono_u) % 2)
    if w < 5:
        smooth_u = mono_u.copy()
    else:
        smooth_u = savgol_filter(mono_u, window_length=w, polyorder=2, mode="interp")
        smooth_u = np.maximum.accumulate(smooth_u)

    # Keep the total cycle count close to the original.
    raw_span = raw_u[-1] - raw_u[0]
    smooth_span = smooth_u[-1] - smooth_u[0]
    if smooth_span > 1e-6:
        smooth_u = (smooth_u - smooth_u[0]) * (raw_span / smooth_span) + raw_u[0]
        smooth_u = np.maximum.accumulate(smooth_u)

    return smooth_u % 1.0


def parse_sheet(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    feat_cols = [c for c in df.columns if c.startswith("Feat_")]
    if not feat_cols:
        return None
    if "Label_Sin" not in df.columns or "Label_Cos" not in df.columns:
        return None
    feat = df[feat_cols].to_numpy(dtype=float)
    sin_vals = df["Label_Sin"].to_numpy(dtype=float)
    cos_vals = df["Label_Cos"].to_numpy(dtype=float)
    return feat, sin_vals, cos_vals


def estimate_quality_score(agg: Dict) -> float:
    score = 1.0
    sat = agg["saturation_ratio"]
    if sat > CONFIG["warn_saturation_ratio"]:
        if sat >= CONFIG["drop_saturation_ratio"]:
            score *= 0.75
        else:
            score *= max(0.82, 1.0 - (sat - CONFIG["warn_saturation_ratio"]) * 2.0)

    levels = agg["phase_levels"]
    if levels < CONFIG["min_phase_levels_warn"]:
        score *= max(0.80, levels / CONFIG["min_phase_levels_warn"])

    cov = agg["phase_coverage"]
    if cov < CONFIG["min_phase_coverage_warn"]:
        score *= max(0.80, cov / CONFIG["min_phase_coverage_warn"])

    cyc = agg["effective_cycles"]
    if cyc < CONFIG["min_effective_cycles_warn"]:
        score *= max(0.80, cyc / CONFIG["min_effective_cycles_warn"])
    return float(np.clip(score, 0.70, 1.10))


def should_drop(agg: Dict) -> Tuple[bool, str]:
    hard_reasons = []
    if agg["cross_sheet_phase_mae"] > CONFIG["max_cross_sheet_phase_mae"]:
        hard_reasons.append("cross_sheet_label_mismatch")
    if agg["phase_coverage"] < CONFIG["min_phase_coverage_hard"]:
        hard_reasons.append("low_phase_coverage")
    if agg["effective_cycles"] < CONFIG["min_effective_cycles_hard"]:
        hard_reasons.append("too_few_cycles")
    if agg["phase_levels"] < CONFIG["min_phase_levels_hard"]:
        hard_reasons.append("very_coarse_phase_levels")
    return len(hard_reasons) > 0, ";".join(hard_reasons) if hard_reasons else "-"


def warn_reasons(agg: Dict) -> str:
    warns = []
    if agg["saturation_ratio"] > CONFIG["warn_saturation_ratio"]:
        warns.append("high_saturation")
    if agg["phase_levels"] < CONFIG["min_phase_levels_warn"]:
        warns.append("coarse_phase_levels")
    if agg["phase_coverage"] < CONFIG["min_phase_coverage_warn"]:
        warns.append("phase_coverage_low_warn")
    if agg["effective_cycles"] < CONFIG["min_effective_cycles_warn"]:
        warns.append("effective_cycles_low_warn")
    return ";".join(warns) if warns else "-"


def clean_xlsx_to_csv(path: str):
    sheet_dict = pd.read_excel(path, sheet_name=None)
    feature_blocks: List[np.ndarray] = []
    ref_phase = None
    sat_ratios = []
    phase_levels = []
    phase_coverages = []
    effective_cycles = []
    cross_maes = []
    n_rows = None

    for _, df in sheet_dict.items():
        parsed = parse_sheet(df)
        if parsed is None:
            continue
        feat, sin_vals, cos_vals = parsed
        phase = wrapped_phase_from_sincos(sin_vals, cos_vals)
        if n_rows is None:
            n_rows = len(phase)
        if len(phase) != n_rows:
            # Length mismatch across sheets is unsafe for concatenation.
            return None, {"drop_reason": "sheet_length_mismatch"}
        pm = phase_metrics(phase)
        sat = float(((feat <= CONFIG["sensor_low_clip"]) | (feat >= CONFIG["sensor_high_clip"])).mean())

        if ref_phase is None:
            ref_phase = phase
            cross = 0.0
        else:
            cross = float(np.mean(circular_diff(ref_phase, phase)))

        feat = np.clip(feat, CONFIG["sensor_low_clip"], CONFIG["sensor_high_clip"])
        feature_blocks.append(feat)
        sat_ratios.append(sat)
        phase_levels.append(pm["phase_levels"])
        phase_coverages.append(pm["phase_coverage"])
        effective_cycles.append(pm["effective_cycles"])
        cross_maes.append(cross)

    if not feature_blocks:
        return None, {"drop_reason": "no_valid_sheet"}

    agg = {
        "saturation_ratio": float(max(sat_ratios)),
        "phase_levels": float(min(phase_levels)),
        "phase_coverage": float(min(phase_coverages)),
        "effective_cycles": float(min(effective_cycles)),
        "cross_sheet_phase_mae": float(max(cross_maes)),
    }
    drop, reason = should_drop(agg)
    if drop:
        agg["drop_reason"] = reason
        return None, agg
    agg["quality_score"] = estimate_quality_score(agg)
    agg["warn_reason"] = warn_reasons(agg)

    phase_clean = reconstruct_phase(np.sin(2 * np.pi * ref_phase), np.cos(2 * np.pi * ref_phase))
    clean_sin = np.sin(2 * np.pi * phase_clean)
    clean_cos = np.cos(2 * np.pi * phase_clean)

    X = np.concatenate(feature_blocks, axis=1)
    cols = [f"Feat_{i + 1}" for i in range(X.shape[1])]
    out_df = pd.DataFrame(X, columns=cols)
    out_df["Label_Sin"] = clean_sin
    out_df["Label_Cos"] = clean_cos
    return out_df, agg


def clean_csv(path: str):
    df = pd.read_csv(path)
    parsed = parse_sheet(df)
    if parsed is None:
        return None, {"drop_reason": "no_valid_label_or_feature_cols"}
    feat, sin_vals, cos_vals = parsed
    phase = wrapped_phase_from_sincos(sin_vals, cos_vals)
    pm = phase_metrics(phase)
    sat = float(((feat <= CONFIG["sensor_low_clip"]) | (feat >= CONFIG["sensor_high_clip"])).mean())
    agg = {
        "saturation_ratio": sat,
        "phase_levels": pm["phase_levels"],
        "phase_coverage": pm["phase_coverage"],
        "effective_cycles": pm["effective_cycles"],
        "cross_sheet_phase_mae": 0.0,
    }
    drop, reason = should_drop(agg)
    if drop:
        agg["drop_reason"] = reason
        return None, agg
    agg["quality_score"] = estimate_quality_score(agg)
    agg["warn_reason"] = warn_reasons(agg)

    feat = np.clip(feat, CONFIG["sensor_low_clip"], CONFIG["sensor_high_clip"])
    phase_clean = reconstruct_phase(sin_vals, cos_vals)
    clean = df.copy()
    feat_cols = [c for c in clean.columns if str(c).strip().startswith("Feat_")]
    clean[feat_cols] = feat
    clean["Label_Sin"] = np.sin(2 * np.pi * phase_clean)
    clean["Label_Cos"] = np.cos(2 * np.pi * phase_clean)
    return clean, agg


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["report_dir"], exist_ok=True)

    files = sorted(glob.glob(os.path.join(CONFIG["input_dir"], "*.xlsx"))) + sorted(
        glob.glob(os.path.join(CONFIG["input_dir"], "*.csv"))
    )
    build_rows = []
    kept = 0
    dropped = 0

    for path in files:
        name = os.path.basename(path)
        try:
            if path.lower().endswith(".xlsx"):
                out_df, agg = clean_xlsx_to_csv(path)
            else:
                out_df, agg = clean_csv(path)
        except Exception as e:
            out_df, agg = None, {"drop_reason": f"read_error:{e}"}

        if out_df is None:
            dropped += 1
            build_rows.append({"file": name, "status": "DROP", **agg})
            continue

        # Save unified clean format as CSV.
        save_name = os.path.splitext(name)[0] + ".csv"
        save_path = os.path.join(CONFIG["output_dir"], save_name)
        out_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        kept += 1
        build_rows.append({"file": name, "status": "KEEP", **agg, "output_file": save_name})

    report_df = pd.DataFrame(build_rows)
    report_csv = os.path.join(CONFIG["report_dir"], "clean_build_report.csv")
    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    summary_path = os.path.join(CONFIG["report_dir"], "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Input files: {len(files)}\n")
        f.write(f"Kept files: {kept}\n")
        f.write(f"Dropped files: {dropped}\n")
        f.write(f"Output dir: {CONFIG['output_dir']}\n")
        f.write(f"Report: {report_csv}\n")

    print("Clean set build completed.")
    print(f"Kept: {kept}, Dropped: {dropped}")
    print(f"Output dir: {CONFIG['output_dir']}")
    print(f"Report: {report_csv}")


if __name__ == "__main__":
    main()
