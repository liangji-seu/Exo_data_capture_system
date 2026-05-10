import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\nightmare\Documents\SRTP")
LSTM_DIR = PROJECT_ROOT / "LSTM"

DEFAULT_RAW_SOURCES = [
    r"D:\nightmare\Documents\SRTP\AllData\202614\raw_data",
]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "AllData" / "Merge_Data_Process"
PREFERRED_PYTHON = LSTM_DIR / ".venv" / "Scripts" / "python.exe"

# -----------------------------------------------------------------------------
# 配置中心（唯一入口）：改 True/False 与路径即可，无需命令行参数。
# -----------------------------------------------------------------------------
PIPELINE_CONFIG: Dict = {
    "raw_sources": DEFAULT_RAW_SOURCES,
    "run_root": str(DEFAULT_RUN_ROOT),
    "subject_start": 7,
    "subject_end": 30,
    "suffix": "",
    # 是否在步骤4完成后自动启动训练
    "run_train": False,
    "train_script": str(LSTM_DIR / "模型9_处理.py"),
    # 是否在步骤2合并后、步骤3审计前，对 merged_dir 内 CSV 做标签帧偏移（逻辑同 repair_label_shift_subset.py 的 REPAIRS）
    "run_label_repair": False,
    "repair_script": str(LSTM_DIR / "repair_label_shift_subset.py"),
}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_tag(path_str: str) -> str:
    name = Path(path_str).name.strip() or "source"
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "source"


def safe_source_tag(path_str: str) -> str:
    """
    Add short hash suffix to avoid collisions for same basename paths.
    Example: raw_data_ab12cd34
    """
    base = safe_tag(path_str)
    key = str(Path(path_str)).replace("\\", "/").lower().encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:8]
    return f"{base}_{digest}"


def normalize_raw_sources(raw_sources: List[str]) -> List[str]:
    """
    Deduplicate input sources while preserving order.
    """
    seen = set()
    out = []
    for src in raw_sources:
        p = str(Path(src))
        key = p.replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def ensure_empty_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_files(path: Path, patterns: List[str]) -> int:
    total = 0
    for p in patterns:
        total += len(list(path.glob(p)))
    return total


def run_ultrasound_stage(ultra_module, raw_sources: List[str], aligned_root: Path, subject_folders: List[str]) -> Dict[str, int]:
    stats = {}
    for src in raw_sources:
        src_path = Path(src)
        if not src_path.exists():
            print(f"[跳过] 原始目录不存在: {src_path}")
            continue

        tag = safe_source_tag(src)
        out_dir = aligned_root / tag
        ensure_empty_dir(out_dir)

        ultra_module.CONFIG["BASE_DIR"] = str(src_path)
        ultra_module.CONFIG["TOTAL_OUTPUT_DIR"] = str(out_dir)
        ultra_module.CONFIG["TARGET_FOLDERS"] = subject_folders

        print(f"\n[步骤1] 处理原始目录: {src_path}")
        print(f"       对齐输出目录: {out_dir}")
        ultra_module.process_all_subjects()

        stats[tag] = count_files(out_dir, ["*.xlsx"])
        print(f"[步骤1] {tag} 生成对齐文件: {stats[tag]}")
    return stats


def merge_aligned_sets(prepare_module, aligned_root: Path, merged_dir: Path) -> Dict[str, int]:
    ensure_empty_dir(merged_dir)
    copied = 0
    skipped = 0

    source_dirs = sorted([p for p in aligned_root.iterdir() if p.is_dir()])
    for src_dir in source_dirs:
        tag = prepare_module.source_tag(str(src_dir))
        for f in src_dir.iterdir():
            if not f.is_file():
                continue
            if not prepare_module.is_trainable_file(f):
                skipped += 1
                continue
            target_name = f"{tag}__{f.name}"
            shutil.copy2(f, merged_dir / target_name)
            copied += 1

    print(f"\n[步骤2] 合并完成: {merged_dir}")
    print(f"[步骤2] 复制文件数: {copied} | 跳过非训练文件数: {skipped}")
    return {"copied": copied, "skipped": skipped}


def run_audit_stage(audit_module, merged_dir: Path, audit_report_dir: Path):
    ensure_empty_dir(audit_report_dir)
    audit_module.CONFIG["input_dir"] = str(merged_dir)
    audit_module.CONFIG["report_dir"] = str(audit_report_dir)
    print(f"\n[步骤3] 标签一致性审计: {merged_dir}")
    audit_module.main()


def run_clean_stage(build_module, merged_dir: Path, clean_dir: Path, clean_report_dir: Path):
    ensure_empty_dir(clean_dir)
    ensure_empty_dir(clean_report_dir)
    build_module.CONFIG["input_dir"] = str(merged_dir)
    build_module.CONFIG["output_dir"] = str(clean_dir)
    build_module.CONFIG["report_dir"] = str(clean_report_dir)
    print(f"\n[步骤4] 清洗并重建训练集: {merged_dir}")
    build_module.main()


def run_label_shift_repair_stage(repair_module, target_dir: Path, repair_report_dir: Path) -> Dict[str, int]:
    """
    在 merged_dir 上原地修改指定 CSV 的 Label_Sin / Label_Cos（按帧平移）。
    修复表来自 repair_module.REPAIRS，与 repair_label_shift_subset.py 一致。
    """
    ensure_empty_dir(repair_report_dir)
    repairs = getattr(repair_module, "REPAIRS", {})
    if not isinstance(repairs, dict) or not repairs:
        print("[步骤2.5][WARN] 未检测到有效 REPAIRS 配置，跳过标签偏移修复。")
        return {"repaired": 0, "missing": 0}

    repaired = 0
    missing = 0
    log_rows = []
    for fname, shift_frames in repairs.items():
        fpath = target_dir / str(fname)
        if not fpath.exists():
            print(f"[步骤2.5][WARN] 修复目标不存在，跳过: {fpath}")
            missing += 1
            continue

        df = pd.read_csv(fpath)
        if "Label_Sin" not in df.columns or "Label_Cos" not in df.columns:
            raise KeyError(f"Label_Sin/Label_Cos 缺失: {fpath}")

        old_sin = df["Label_Sin"].to_numpy(dtype=np.float64)
        old_cos = df["Label_Cos"].to_numpy(dtype=np.float64)
        shift_i = int(shift_frames)
        new_sin = np.roll(old_sin, shift_i)
        new_cos = np.roll(old_cos, shift_i)

        norm = np.sqrt(new_sin * new_sin + new_cos * new_cos)
        norm = np.where(norm < 1e-8, 1.0, norm)
        df["Label_Sin"] = (new_sin / norm).astype(np.float64)
        df["Label_Cos"] = (new_cos / norm).astype(np.float64)
        df.to_csv(fpath, index=False, encoding="utf-8-sig")

        repaired += 1
        log_rows.append(
            {
                "file": str(fname),
                "shift_frames": shift_i,
                "n_rows": int(len(df)),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    log_path = repair_report_dir / "label_shift_repair_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "shift_frames", "n_rows", "timestamp"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\n[步骤2.5] 标签偏移修复完成: repaired={repaired}, missing={missing}")
    print(f"[步骤2.5] 修复日志: {log_path}")
    return {"repaired": repaired, "missing": missing}


def write_manifest(run_dir: Path, manifest: Dict):
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    quick_txt = run_dir / "quick_use_paths.txt"
    with open(quick_txt, "w", encoding="utf-8") as f:
        f.write(f"run_dir={run_dir}\n")
        f.write(f"merged_trainable_dir={manifest['paths']['merged_trainable_dir']}\n")
        f.write(f"clean_dataset_dir={manifest['paths']['clean_dataset_dir']}\n")
        f.write(f"audit_report_dir={manifest['paths']['audit_report_dir']}\n")
        f.write(f"clean_build_report_dir={manifest['paths']['clean_build_report_dir']}\n")
        f.write(f"audit_main_csv={manifest['paths']['audit_main_csv']}\n")
        f.write(f"clean_build_csv={manifest['paths']['clean_build_csv']}\n")
        f.write(f"label_shift_repair_report_dir={manifest['paths']['label_shift_repair_report_dir']}\n")
        f.write(f"label_shift_repair_csv={manifest['paths']['label_shift_repair_csv']}\n")


def print_runtime_info():
    print(f"[RUNTIME] Python: {sys.executable}")
    for dep in ("openpyxl", "matplotlib", "pandas", "numpy"):
        try:
            __import__(dep)
            print(f"[RUNTIME] 依赖可用: {dep}")
        except Exception as e:
            print(f"[RUNTIME][WARN] 依赖不可用: {dep} ({e})")


def ensure_preferred_python():
    """
    Auto-relaunch the pipeline with project .venv Python to avoid
    dependency mismatch (e.g., openpyxl/matplotlib missing in system Python).
    """
    if os.environ.get("MERGE_PIPELINE_REEXEC") == "1":
        return
    if not PREFERRED_PYTHON.exists():
        return

    current = Path(sys.executable).resolve()
    preferred = PREFERRED_PYTHON.resolve()
    if current == preferred:
        return

    print(f"[INFO] 当前解释器: {current}")
    print(f"[INFO] 自动切换到项目虚拟环境解释器: {preferred}")
    env = os.environ.copy()
    env["MERGE_PIPELINE_REEXEC"] = "1"
    cmd = [str(preferred), str(Path(__file__).resolve()), *sys.argv[1:]]
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)


def run_training_stage(train_script: str, clean_dir: Path, clean_report_dir: Path, run_dir: Path):
    train_script_path = Path(train_script)
    if not train_script_path.exists():
        raise FileNotFoundError(f"训练脚本不存在: {train_script_path}")

    train_save_dir = run_dir / "05_training_logs"
    train_save_dir.mkdir(parents=True, exist_ok=True)

    quality_csv = clean_report_dir / "clean_build_report.csv"
    try:
        train_text = train_script_path.read_text(encoding="utf-8")
    except Exception:
        train_text = ""
    required_env_keys = ["M9_DATA_DIR", "M9_QUALITY_REPORT_PATH", "M9_SAVE_DIR"]
    missing_keys = [k for k in required_env_keys if k not in train_text]
    if missing_keys:
        print(f"[步骤5][WARN] 训练脚本可能未显式处理环境变量覆盖: {missing_keys}")
        print("[步骤5][WARN] 将继续执行，但请确认训练脚本确实使用本次流水线输出目录。")

    env = os.environ.copy()
    env["M9_DATA_DIR"] = str(clean_dir)
    env["M9_QUALITY_REPORT_PATH"] = str(quality_csv)
    env["M9_SAVE_DIR"] = str(train_save_dir)

    cmd = [sys.executable, str(train_script_path)]
    print(f"\n[步骤5] 启动训练: {' '.join(cmd)}")
    print(f"[步骤5] M9_DATA_DIR={clean_dir}")
    print(f"[步骤5] M9_QUALITY_REPORT_PATH={quality_csv}")
    print(f"[步骤5] M9_SAVE_DIR={train_save_dir}")

    start = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = round(time.time() - start, 2)
    if proc.returncode != 0:
        raise RuntimeError(f"训练失败，退出码={proc.returncode}")
    return {"train_seconds": elapsed, "train_save_dir": str(train_save_dir), "train_script": str(train_script_path)}


def main():
    ensure_preferred_python()
    print_runtime_info()
    cfg = PIPELINE_CONFIG

    if int(cfg["subject_start"]) > int(cfg["subject_end"]):
        raise ValueError("PIPELINE_CONFIG['subject_start'] 不能大于 subject_end")

    raw_sources = normalize_raw_sources([str(Path(p)) for p in cfg["raw_sources"]])
    subject_folders = [str(i) for i in range(int(cfg["subject_start"]), int(cfg["subject_end"]) + 1)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pipeline_run_{ts}{cfg['suffix']}"
    run_root = Path(cfg["run_root"])
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    aligned_root = run_dir / "01_aligned_per_source"
    merged_dir = run_dir / "02_merged_trainable"
    reports_root = run_dir / "03_reports"
    audit_report_dir = reports_root / "label_consistency_audit"
    clean_report_dir = reports_root / "clean_set_build"
    repair_report_dir = reports_root / "label_shift_repair"
    clean_dir = run_dir / "04_clean_dataset"

    aligned_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    ultra_module = load_module(LSTM_DIR / "超声数据处理.py", "ultra_process_pipeline")
    prepare_module = load_module(LSTM_DIR / "prepare_joint_training_set.py", "prepare_joint_pipeline")
    audit_module = load_module(LSTM_DIR / "label_consistency_audit.py", "label_audit_pipeline")
    build_module = load_module(LSTM_DIR / "build_clean_training_set_v2.py", "clean_set_pipeline")
    repair_module = None
    if bool(cfg.get("run_label_repair", False)):
        repair_path = Path(str(cfg.get("repair_script", "")))
        if not repair_path.exists():
            raise FileNotFoundError(f"标签偏移修复脚本不存在: {repair_path}")
        repair_module = load_module(repair_path, "label_shift_repair_pipeline")

    t0 = time.time()
    step_t = {}

    s = time.time()
    aligned_stats = run_ultrasound_stage(ultra_module, raw_sources, aligned_root, subject_folders)
    step_t["ultrasound_align_seconds"] = round(time.time() - s, 2)
    aligned_total = int(sum(aligned_stats.values())) if aligned_stats else 0
    if aligned_total <= 0:
        raise RuntimeError("步骤1未产出任何对齐文件，请检查原始数据路径、依赖和对齐参数。")

    s = time.time()
    merge_stats = merge_aligned_sets(prepare_module, aligned_root, merged_dir)
    step_t["merge_seconds"] = round(time.time() - s, 2)
    if int(merge_stats.get("copied", 0)) <= 0:
        raise RuntimeError("步骤2合并后无训练文件，已中止。")

    repair_stats = {"repaired": 0, "missing": 0}
    if bool(cfg.get("run_label_repair", False)):
        s = time.time()
        repair_stats = run_label_shift_repair_stage(repair_module, merged_dir, repair_report_dir)
        step_t["label_shift_repair_seconds"] = round(time.time() - s, 2)

    s = time.time()
    run_audit_stage(audit_module, merged_dir, audit_report_dir)
    step_t["audit_seconds"] = round(time.time() - s, 2)

    s = time.time()
    run_clean_stage(build_module, merged_dir, clean_dir, clean_report_dir)
    step_t["clean_build_seconds"] = round(time.time() - s, 2)
    clean_csv_count = count_files(clean_dir, ["*.csv"])
    if clean_csv_count <= 0:
        raise RuntimeError("步骤4 clean 数据集为空，已中止。")

    training_info = None
    if bool(cfg.get("run_train", False)):
        s = time.time()
        training_info = run_training_stage(str(cfg["train_script"]), clean_dir, clean_report_dir, run_dir)
        step_t["train_seconds"] = round(time.time() - s, 2)

    total_sec = round(time.time() - t0, 2)

    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_sources": raw_sources,
        "subject_folders": subject_folders,
        "stats": {
            "aligned_files_per_source": aligned_stats,
            "merged_copy_stats": merge_stats,
            "label_shift_repair_enabled": bool(cfg.get("run_label_repair", False)),
            "label_shift_repair_stats": repair_stats,
            "clean_csv_count": clean_csv_count,
            "train_enabled": bool(cfg.get("run_train", False)),
            "training_info": training_info,
            "timings_seconds": step_t,
            "total_seconds": total_sec,
        },
        "paths": {
            "run_dir": str(run_dir),
            "aligned_root": str(aligned_root),
            "merged_trainable_dir": str(merged_dir),
            "audit_report_dir": str(audit_report_dir),
            "clean_build_report_dir": str(clean_report_dir),
            "label_shift_repair_report_dir": str(repair_report_dir),
            "clean_dataset_dir": str(clean_dir),
            "audit_main_csv": str(audit_report_dir / "label_consistency_audit.csv"),
            "clean_build_csv": str(clean_report_dir / "clean_build_report.csv"),
            "label_shift_repair_csv": str(repair_report_dir / "label_shift_repair_log.csv"),
            "training_save_dir": str(run_dir / "05_training_logs") if bool(cfg.get("run_train", False)) else "",
        },
    }
    write_manifest(run_dir, manifest)

    print("\n" + "=" * 72)
    print("完整数据处理流水线执行完成")
    print(f"运行目录: {run_dir}")
    print(f"clean 数据集文件数: {clean_csv_count}")
    print(f"总耗时: {total_sec} 秒")
    print(f"清单文件: {run_dir / 'run_manifest.json'}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] 流水线失败: {e}")
        sys.exit(1)
