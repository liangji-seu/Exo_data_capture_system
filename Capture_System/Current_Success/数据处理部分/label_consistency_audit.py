import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CONFIG = {
    "input_dir": r"D:\nightmare\Documents\SRTP\AllData\2026_1_26\Processed_Final_Training_Set_1_26_1",
    "report_dir": r"D:\nightmare\Documents\SRTP\LSTM\training_logs\label_audit_1_26_1",
    # quality thresholds
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
            "phase_step_median": 0.0,
            "phase_coverage": 0.0,
            "effective_cycles": 0.0,
            "backward_ratio": 0.0,
        }
    unique_vals = np.sort(np.unique(np.round(phase, 6)))
    diffs = np.diff(unique_vals)
    unwrapped = np.unwrap(phase * 2 * np.pi) / (2 * np.pi)
    du = np.diff(unwrapped)
    return {
        "phase_levels": float(len(unique_vals)),
        "phase_step_median": float(np.median(diffs)) if len(diffs) else 0.0,
        "phase_coverage": float(np.quantile(phase, 0.95) - np.quantile(phase, 0.05)),
        "effective_cycles": float(unwrapped[-1] - unwrapped[0]),
        "backward_ratio": float(np.mean(du < -0.05)) if len(du) else 0.0,
    }


def parse_domain(file_name: str) -> str:
    if "__" in file_name:
        return file_name.split("__", 1)[0]
    return "unknown"


def sheet_features_and_labels(df: pd.DataFrame):
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


def analyze_xlsx(path: str) -> Tuple[Dict, List[Dict]]:
    sheet_dict = pd.read_excel(path, sheet_name=None)
    per_sheet = []
    ref_phase = None
    min_len = None

    for sheet_name, df in sheet_dict.items():
        parsed = sheet_features_and_labels(df)
        if parsed is None:
            continue
        feat, sin_vals, cos_vals = parsed
        phase = wrapped_phase_from_sincos(sin_vals, cos_vals)
        sat_ratio = float(((feat <= CONFIG["sensor_low_clip"]) | (feat >= CONFIG["sensor_high_clip"])).mean())
        pm = phase_metrics(phase)
        if ref_phase is None:
            ref_phase = phase
            min_len = len(phase)
            cross_mae = 0.0
        else:
            n = min(min_len, len(phase))
            cross_mae = float(np.mean(circular_diff(ref_phase[:n], phase[:n])))

        per_sheet.append(
            {
                "sheet": sheet_name,
                "n_samples": len(phase),
                "n_features": feat.shape[1],
                "saturation_ratio": sat_ratio,
                "cross_sheet_phase_mae": cross_mae,
                **pm,
            }
        )

    if not per_sheet:
        return {}, []

    agg = {
        "n_sheets": len(per_sheet),
        "n_samples": min(x["n_samples"] for x in per_sheet),
        "n_features_total": int(sum(x["n_features"] for x in per_sheet)),
        "saturation_ratio_max": float(max(x["saturation_ratio"] for x in per_sheet)),
        "phase_levels_min": float(min(x["phase_levels"] for x in per_sheet)),
        "phase_step_median_max": float(max(x["phase_step_median"] for x in per_sheet)),
        "phase_coverage_min": float(min(x["phase_coverage"] for x in per_sheet)),
        "effective_cycles_min": float(min(x["effective_cycles"] for x in per_sheet)),
        "cross_sheet_phase_mae_max": float(max(x["cross_sheet_phase_mae"] for x in per_sheet)),
    }
    return agg, per_sheet


def analyze_csv(path: str) -> Tuple[Dict, List[Dict]]:
    df = pd.read_csv(path)
    parsed = sheet_features_and_labels(df)
    if parsed is None:
        return {}, []
    feat, sin_vals, cos_vals = parsed
    phase = wrapped_phase_from_sincos(sin_vals, cos_vals)
    sat_ratio = float(((feat <= CONFIG["sensor_low_clip"]) | (feat >= CONFIG["sensor_high_clip"])).mean())
    pm = phase_metrics(phase)
    agg = {
        "n_sheets": 1,
        "n_samples": len(phase),
        "n_features_total": int(feat.shape[1]),
        "saturation_ratio_max": sat_ratio,
        "phase_levels_min": pm["phase_levels"],
        "phase_step_median_max": pm["phase_step_median"],
        "phase_coverage_min": pm["phase_coverage"],
        "effective_cycles_min": pm["effective_cycles"],
        "cross_sheet_phase_mae_max": 0.0,
    }
    return agg, [{"sheet": "csv", **agg}]


def estimate_quality_score(agg: Dict) -> float:
    score = 1.0
    sat = agg["saturation_ratio_max"]
    if sat > CONFIG["warn_saturation_ratio"]:
        if sat >= CONFIG["drop_saturation_ratio"]:
            score *= 0.75
        else:
            score *= max(0.82, 1.0 - (sat - CONFIG["warn_saturation_ratio"]) * 2.0)

    levels = agg["phase_levels_min"]
    if levels < CONFIG["min_phase_levels_warn"]:
        score *= max(0.80, levels / CONFIG["min_phase_levels_warn"])

    cov = agg["phase_coverage_min"]
    if cov < CONFIG["min_phase_coverage_warn"]:
        score *= max(0.80, cov / CONFIG["min_phase_coverage_warn"])

    cyc = agg["effective_cycles_min"]
    if cyc < CONFIG["min_effective_cycles_warn"]:
        score *= max(0.80, cyc / CONFIG["min_effective_cycles_warn"])

    return float(np.clip(score, 0.70, 1.10))


def decide_status(agg: Dict) -> Tuple[str, str, float]:
    hard_reasons = []
    warn_reasons = []

    # Hard failures: these are difficult to repair reliably.
    if agg["cross_sheet_phase_mae_max"] > CONFIG["max_cross_sheet_phase_mae"]:
        hard_reasons.append("cross_sheet_label_mismatch")
    if agg["phase_coverage_min"] < CONFIG["min_phase_coverage_hard"]:
        hard_reasons.append("low_phase_coverage")
    if agg["effective_cycles_min"] < CONFIG["min_effective_cycles_hard"]:
        hard_reasons.append("too_few_cycles")
    if agg["phase_levels_min"] < CONFIG["min_phase_levels_hard"]:
        hard_reasons.append("very_coarse_phase_levels")

    # Soft issues: can be fixed by label reconstruction / feature clipping.
    if agg["phase_levels_min"] < CONFIG["min_phase_levels_warn"]:
        warn_reasons.append("coarse_phase_levels")
    if agg["saturation_ratio_max"] > CONFIG["warn_saturation_ratio"]:
        warn_reasons.append("high_saturation")
    if agg["saturation_ratio_max"] > CONFIG["drop_saturation_ratio"]:
        warn_reasons.append("very_high_saturation")
    if agg["phase_coverage_min"] < CONFIG["min_phase_coverage_warn"]:
        warn_reasons.append("phase_coverage_low_warn")
    if agg["effective_cycles_min"] < CONFIG["min_effective_cycles_warn"]:
        warn_reasons.append("effective_cycles_low_warn")

    quality_score = estimate_quality_score(agg)
    if hard_reasons:
        return "DROP", ";".join(hard_reasons + warn_reasons if warn_reasons else hard_reasons), quality_score
    if warn_reasons:
        return "WARN", ";".join(warn_reasons), quality_score
    return "KEEP", "-", quality_score


def main():
    os.makedirs(CONFIG["report_dir"], exist_ok=True)
    files = sorted(glob.glob(os.path.join(CONFIG["input_dir"], "*.xlsx"))) + sorted(
        glob.glob(os.path.join(CONFIG["input_dir"], "*.csv"))
    )
    rows = []
    sheet_rows = []

    for path in files:
        name = os.path.basename(path)
        try:
            if path.lower().endswith(".xlsx"):
                agg, per_sheet = analyze_xlsx(path)
            else:
                agg, per_sheet = analyze_csv(path)
            if not agg:
                continue
            status, reason, quality_score = decide_status(agg)
            row = {
                "file": name,
                "domain": parse_domain(name),
                "status": status,
                "reason": reason,
                "recommended_weight": quality_score,
                **agg,
            }
            rows.append(row)
            for s in per_sheet:
                sheet_rows.append({"file": name, **s})
        except Exception as e:
            rows.append({"file": name, "domain": parse_domain(name), "status": "DROP", "reason": f"read_error:{e}"})

    audit_df = pd.DataFrame(rows)
    sheet_df = pd.DataFrame(sheet_rows)
    audit_csv = os.path.join(CONFIG["report_dir"], "label_consistency_audit.csv")
    sheet_csv = os.path.join(CONFIG["report_dir"], "label_consistency_audit_per_sheet.csv")
    audit_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    sheet_df.to_csv(sheet_csv, index=False, encoding="utf-8-sig")

    summary_path = os.path.join(CONFIG["report_dir"], "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total files analyzed: {len(audit_df)}\n")
        if len(audit_df):
            f.write(f"KEEP: {(audit_df['status'] == 'KEEP').sum()}\n")
            f.write(f"WARN: {(audit_df['status'] == 'WARN').sum()}\n")
            f.write(f"DROP: {(audit_df['status'] == 'DROP').sum()}\n")
            f.write("\nTop DROP reasons:\n")
            drops = audit_df[audit_df["status"] == "DROP"]["reason"].value_counts()
            for k, v in drops.items():
                f.write(f"- {k}: {v}\n")

    print("Audit completed.")
    print(f"Main report: {audit_csv}")
    print(f"Per-sheet report: {sheet_csv}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
