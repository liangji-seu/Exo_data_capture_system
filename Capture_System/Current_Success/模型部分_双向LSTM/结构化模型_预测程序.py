import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt  # 新增：引入 matplotlib 用于绘图

CONFIG = {
    # 训练输出目录（含 best_model_overall.keras / best_scaler_overall.pkl / best_pca_overall.pkl）
    # 填好后可直接运行本脚本而无需命令行传 --model-dir；命令行仍会覆盖此项。
    "MODEL_DIR": r"D:\nightmare\Documents\SRTP\LSTM\training_logs\ultr_fusion_model_9_auto_pro_1",
    # 例: r"D:\nightmare\Documents\SRTP\LSTM\training_logs\ultr_fusion_model_9_auto_pro_1"

    # 是否加载原始数据并运行 merge_data_pipeline.py 生成 clean_dataset_dir
    # - True: 使用 RAW_SOURCES + pipeline-run-root + subject range 生成 clean 数据后预测
    # - False: 直接使用 PROCESSED_DATA_DIR（由外部提前生成）
    "LOAD_RAW_DATA": False,

    # 当 LOAD_RAW_DATA=True 时：
    "RAW_SOURCES": [
        # r"D:\path\to\raw_data_dir1",
        # r"D:\path\to\raw_data_dir2",
    ],
    "PIPELINE_RUN_ROOT": str(Path(r"D:\nightmare\Documents\SRTP\AllData\Merge_Data_Process")),
    "SUBJECT_START": 7,
    "SUBJECT_END": 30,
    "PIPELINE_SUFFIX": "",
    "RUN_LABEL_REPAIR": False,

    # 当 LOAD_RAW_DATA=False 时：填「目录」（04_clean_dataset）或单个 .csv/.xlsx 文件路径。
    "PROCESSED_DATA_DIR": r"D:\nightmare\Documents\SRTP\AllData\Prediction_Data",

    # 后处理策略（需要与模型9_处理.py 候选兼容）
    "SMOOTHING_STRATEGY": "perfect_sawtooth",

    # 是否在预测结束后自动绘制并保存曲线图
    "PLOT_RESULTS": True,
}


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_pickle_main_aliases_from_train_module(train_module):
    """训练时若以 `python 模型9_处理.py` 等方式运行，joblib 会把 scaler 记成 `__main__.AdaptiveScaler`。
    预测脚本自己的 __main__ 里没有该类，反序列化会失败。将训练脚本里的类挂到当前 __main__ 上即可。"""
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    if hasattr(train_module, "AdaptiveScaler") and not hasattr(main_mod, "AdaptiveScaler"):
        setattr(main_mod, "AdaptiveScaler", train_module.AdaptiveScaler)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_latest_pipeline_run(run_root: Path, suffix: str = "") -> Path:
    if not run_root.exists():
        raise FileNotFoundError(f"run_root not found: {run_root}")
    candidates = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("pipeline_run_")]
    if suffix:
        candidates = [p for p in candidates if p.name.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"No pipeline_run_* directory found under: {run_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_merge_pipeline_and_get_clean_dir(
        merge_script: Path,
        raw_sources,
        run_root: Path,
        subject_start: int,
        subject_end: int,
        suffix: str = "",
        run_label_repair: bool = False,
        preferred_python: Path | None = None,
) -> Path:
    ensure_dir(run_root)
    cmd = [
        str(preferred_python or sys.executable),
        str(merge_script),
        "--raw-sources",
        *[str(s) for s in raw_sources],
        "--run-root",
        str(run_root),
        "--subject-start",
        str(subject_start),
        "--subject-end",
        str(subject_end),
        "--suffix",
        suffix,
    ]
    if run_label_repair:
        cmd.append("--run-label-repair")

    print("[Predict] Running merge_data_pipeline ...")
    print("[Predict] CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    run_dir = find_latest_pipeline_run(run_root, suffix=suffix)
    clean_dir = run_dir / "04_clean_dataset"
    if not clean_dir.exists():
        raise FileNotFoundError(f"clean_dir not found: {clean_dir}")
    print(f"[Predict] Using clean_dataset_dir: {clean_dir}")
    return clean_dir


def load_trial_features_labels(file_path: Path, base_module):
    """
    Load one trial.
    Returns:
      features: np.ndarray [T, F]
      true_phase: np.ndarray[T] or None
      trial_name: str (basename without ext)
    """
    trial_name = file_path.stem

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        sheet_dict = pd.read_excel(file_path, sheet_name=None)
        # concat all sheets' features; prefer first sheet containing labels
        feats_all = []
        sin_vals = None
        cos_vals = None
        for _, sdf in sheet_dict.items():
            sdf.columns = sdf.columns.str.strip()
            feat_cols = [c for c in sdf.columns if str(c).startswith("Feat_")]
            if feat_cols:
                feats_all.append(sdf[feat_cols].values)
            if sin_vals is None and "Label_Sin" in sdf.columns:
                sin_vals = sdf["Label_Sin"].values
                if "Label_Cos" in sdf.columns:
                    cos_vals = sdf["Label_Cos"].values
                else:
                    cos_vals = base_module.robust_cos_reconstruction(sin_vals)
        if not feats_all:
            raise ValueError(f"No Feat_ columns found in any sheet: {file_path}")
        features = np.concatenate(feats_all, axis=1)
        if sin_vals is not None:
            true_rad = np.arctan2(sin_vals, cos_vals)
            true_phase = (true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        else:
            true_phase = None
        return features.astype(np.float32), true_phase.astype(
            np.float32) if true_phase is not None else None, trial_name
    else:
        raise ValueError(f"Unsupported file: {file_path}")

    df.columns = df.columns.str.strip()
    feat_cols = [c for c in df.columns if str(c).startswith("Feat_")]
    if not feat_cols:
        raise ValueError(f"No Feat_ columns found: {file_path}")
    features = df[feat_cols].values.astype(np.float32)

    true_phase = None
    if "Label_Sin" in df.columns:
        sin_vals = df["Label_Sin"].values.astype(np.float32)
        if "Label_Cos" in df.columns:
            cos_vals = df["Label_Cos"].values.astype(np.float32)
        else:
            cos_vals = base_module.robust_cos_reconstruction(sin_vals)
        true_rad = np.arctan2(sin_vals, cos_vals)
        true_phase = ((true_rad + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)).astype(np.float32)
    return features, true_phase, trial_name


def create_sequences_only(features: np.ndarray, time_steps: int) -> np.ndarray:
    """
    features: [T, F]
    returns: X_seq [N, time_steps, F]
    """
    if len(features) <= time_steps:
        return np.empty((0, time_steps, features.shape[1]), dtype=np.float32)
    xs = []
    for i in range(len(features) - time_steps):
        xs.append(features[i: i + time_steps])
    return np.asarray(xs, dtype=np.float32)


def plot_prediction(df: pd.DataFrame, trial_name: str, output_dir: Path, mae: float = None):
    """
    绘制预测相位与真实相位的对比曲线图。
    处理了相位循环时的跳变问题，使折线图更美观。
    """
    plt.figure(figsize=(14, 5))

    def plot_phase_with_breaks(x, y, label, color, linestyle='-'):
        """如果相位变化 > 0.5 (即发生 0 <-> 1 循环跳变)，插入 NaN 断开连线，防止垂直跳变线"""
        x_plot = x.astype(float).copy()
        y_plot = y.astype(float).copy()

        valid_idx = ~np.isnan(y_plot)
        if not np.any(valid_idx):
            return

        diffs = np.zeros_like(y_plot)
        diffs[1:] = np.abs(y_plot[1:] - y_plot[:-1])
        break_indices = np.where(diffs > 0.5)[0]

        if len(break_indices) > 0:
            x_plot = np.insert(x_plot, break_indices, np.nan)
            y_plot = np.insert(y_plot, break_indices, np.nan)

        plt.plot(x_plot, y_plot, label=label, color=color, linestyle=linestyle, alpha=0.8)

    # 绘制真实相位和预测相位
    if "True_Phase" in df.columns:
        plot_phase_with_breaks(df["Time_Step"].values, df["True_Phase"].values, "True Phase", "lightcoral", "--")
    plot_phase_with_breaks(df["Time_Step"].values, df["Predicted_Phase"].values, "Predicted Phase", "royalblue", "-")

    # 构建标题
    title = f"Phase Prediction: {trial_name}"
    if mae is not None and not np.isnan(mae):
        title += f" | MAE: {mae * 100:.2f}%"

    plt.title(title)
    plt.xlabel("Time Step (Frames)")
    plt.ylabel("Phase (0.0 - 1.0)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 保存图片
    out_img = output_dir / f"{trial_name}_plot.png"
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"[Plot] Saved plot -> {out_img}")


def predict_one_trial(
        file_path: Path,
        base_module,
        post_module,
        model,
        scaler,
        pca,
        time_steps: int,
        smoothing_strategy: str,
        output_dir: Path,
        plot_results: bool = True,  # 新增：控制是否绘图
):
    features, true_phase, trial_name = load_trial_features_labels(file_path, base_module)
    if features.shape[0] <= time_steps:
        print(f"[Predict] Skip (too short): {file_path} (T={features.shape[0]})")
        return None

    # Keep the same preprocessing as training: scaler -> PCA -> sequences.
    f_scaled = scaler.transform(features)
    f_pca = pca.transform(f_scaled)
    X_seq = create_sequences_only(f_pca, time_steps)

    preds_raw = model.predict(X_seq, verbose=0)
    pred_phases = post_module.smooth_predictions(
        base_module,
        preds_raw,
        strategy=smoothing_strategy,
    )
    pred_phases = np.asarray(pred_phases, dtype=np.float32)  # [N]

    # Pad to original length: first `time_steps` frames have no prediction.
    padded_pred = np.full((len(features),), np.nan, dtype=np.float32)
    padded_pred[time_steps:] = pred_phases

    out_df = pd.DataFrame(
        {
            "Time_Step": np.arange(len(features), dtype=np.int32),
            "Predicted_Phase": padded_pred,
        }
    )

    if true_phase is not None and len(true_phase) == len(features):
        padded_true = np.asarray(true_phase, dtype=np.float32)
        out_df["True_Phase"] = padded_true
        # MAE on predicted portion only
        mae = post_module.circular_mae(padded_pred[time_steps:], padded_true[time_steps:])
        out_df["Trial_MAE"] = np.nan
        out_df.loc[time_steps:, "Trial_MAE"] = mae
    else:
        mae = None

    ensure_dir(output_dir)
    out_csv = output_dir / f"{trial_name}_predicted.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Predict] {trial_name}: saved -> {out_csv}")
    if mae is not None:
        print(f"[Predict] {trial_name}: MAE(no_lag)={mae * 100:.2f}%")

    # 如果开启了绘图，就在这调用绘图函数
    if plot_results:
        plot_prediction(out_df, trial_name, output_dir, mae)

    return mae


def main():
    parser = argparse.ArgumentParser(description="模型9_预测程序（支持原始数据->pipeline->推理以及自动绘图）")

    parser.add_argument(
        "--model-dir",
        default=None,
        help="训练输出目录（含 best_model_overall.keras 等）。可省略：使用 CONFIG['MODEL_DIR']。",
    )
    # 下面两个参数仍保留，用于临时覆盖 CONFIG；但开关推荐用 CONFIG['LOAD_RAW_DATA']
    parser.add_argument("--raw-sources", nargs="*", default=[], help="（可选覆盖）原始超声数据目录列表")
    parser.add_argument("--processed-data-dir", default="", help="（可选覆盖）已准备好的 clean_dataset_dir 目录")
    parser.add_argument("--pipeline-run-root",
                        default=str(Path(r"D:\nightmare\Documents\SRTP\AllData\Merge_Data_Process")),
                        help="pipeline 运行输出根目录")
    parser.add_argument("--subject-start", type=int, default=7)
    parser.add_argument("--subject-end", type=int, default=30)
    parser.add_argument("--pipeline-suffix", default="", help="pipeline run 后缀（用于区分不同实验）")
    parser.add_argument("--run-label-repair", action="store_true", help="是否运行标签偏移修复步骤（可选）")

    parser.add_argument("--smoothing-strategy", default=CONFIG["SMOOTHING_STRATEGY"],
                        help="后处理策略：需与模型9_处理兼容")
    parser.add_argument("--output-dir", default="", help="输出目录（默认输出到 model-dir 的预测子目录）")

    # 增加命令行强制控制绘图的开关
    parser.add_argument("--plot", action="store_true", help="绘制并保存预测结果对比图 (覆盖CONFIG)")
    parser.add_argument("--no-plot", action="store_true", help="禁用绘制预测结果图 (覆盖CONFIG)")

    args = parser.parse_args()

    # 解析绘图开关：命令行优先，未指定则用 CONFIG
    plot_results = CONFIG.get("PLOT_RESULTS", True)
    if args.plot:
        plot_results = True
    elif args.no_plot:
        plot_results = False

    model_dir_str = (args.model_dir or "").strip() or str(CONFIG.get("MODEL_DIR", "")).strip()
    if not model_dir_str:
        raise SystemExit(
            "必须指定模型目录：\n"
            "  1) 命令行: python 结构化模型_预测程序.py --model-dir \"你的training_logs目录\"\n"
            "  2) 或在脚本顶部 CONFIG['MODEL_DIR'] 中填写同一路径。"
        )

    project_root = Path(r"D:\nightmare\Documents\SRTP")
    lstm_dir = project_root / "LSTM"
    merge_script = lstm_dir / "merge_data_pipeline.py"

    m9_base = load_module(lstm_dir / "模型9_增强.py", "m9_base")
    m9_post = load_module(lstm_dir / "模型9_处理.py", "m9_post")
    patch_pickle_main_aliases_from_train_module(m9_post)

    model_dir = Path(model_dir_str)
    model_path = model_dir / "best_model_overall.keras"
    scaler_path = model_dir / "best_scaler_overall.pkl"
    pca_path = model_dir / "best_pca_overall.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"best_model_overall.keras not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"best_scaler_overall.pkl not found: {scaler_path}")
    if not pca_path.exists():
        raise FileNotFoundError(f"best_pca_overall.pkl not found: {pca_path}")

    print("[Predict] Loading scaler/pca/model ...")
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Infer time_steps from model input shape.
    # model.input_shape: (None, time_steps, feature_dim)
    input_shape = model.input_shape
    if not input_shape or len(input_shape) < 3:
        raise RuntimeError(f"Unexpected model input_shape: {input_shape}")
    time_steps = int(input_shape[1])

    output_dir = Path(args.output_dir) if args.output_dir else (model_dir / "predictions")
    ensure_dir(output_dir)

    # Prepare clean data directory.
    load_raw = bool(CONFIG.get("LOAD_RAW_DATA", False))
    if not load_raw:
        processed_dir = args.processed_data_dir or CONFIG.get("PROCESSED_DATA_DIR", "")
        if not processed_dir:
            raise ValueError("LOAD_RAW_DATA=False 时必须配置 CONFIG['PROCESSED_DATA_DIR'] 或传入 --processed-data-dir。")
        clean_dir = Path(processed_dir)
        if not clean_dir.exists():
            raise FileNotFoundError(f"processed-data-dir not found: {clean_dir}")
        if clean_dir.is_file() and clean_dir.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            raise ValueError(
                f"PROCESSED_DATA_DIR 应为 clean 数据「目录」或单个 .csv/.xlsx 文件，当前不是有效数据文件: {clean_dir}"
            )
    else:
        raw_sources = args.raw_sources or CONFIG.get("RAW_SOURCES", [])
        if not raw_sources:
            raise ValueError("LOAD_RAW_DATA=True 时必须配置 CONFIG['RAW_SOURCES'] 或传入 --raw-sources。")
        clean_dir = run_merge_pipeline_and_get_clean_dir(
            merge_script=merge_script,
            raw_sources=raw_sources,
            run_root=Path(args.pipeline_run_root or CONFIG.get("PIPELINE_RUN_ROOT", "")),
            subject_start=args.subject_start if args.subject_start is not None else int(CONFIG.get("SUBJECT_START", 7)),
            subject_end=args.subject_end if args.subject_end is not None else int(CONFIG.get("SUBJECT_END", 30)),
            suffix=args.pipeline_suffix if args.pipeline_suffix is not None else str(CONFIG.get("PIPELINE_SUFFIX", "")),
            run_label_repair=bool(args.run_label_repair) if any([args.run_label_repair]) else bool(
                CONFIG.get("RUN_LABEL_REPAIR", False)),
            preferred_python=lstm_dir / ".venv" / "Scripts" / "python.exe",
        )

    # Find candidate files（目录则扫描；若指向单个 trial 文件则只预测该文件）。
    if clean_dir.is_file():
        candidates = [clean_dir]
    else:
        candidates = sorted([
            p
            for p in clean_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        ]
        )
    if not candidates:
        raise FileNotFoundError(f"No .csv/.xlsx files found under clean_dir: {clean_dir}")

    print(f"[Predict] Found {len(candidates)} trial files.")
    maes = []
    for fp in candidates:
        mae = predict_one_trial(
            file_path=fp,
            base_module=m9_base,
            post_module=m9_post,
            model=model,
            scaler=scaler,
            pca=pca,
            time_steps=time_steps,
            smoothing_strategy=args.smoothing_strategy,
            output_dir=output_dir,
            plot_results=plot_results,  # 传递绘图开关
        )
        if mae is not None:
            maes.append(float(mae))

    if maes:
        print(f"[Predict] Done. Mean MAE(no_lag)={np.mean(maes) * 100:.2f}% over {len(maes)} labeled trials.")
    else:
        print("[Predict] Done. No labeled trials found; only predictions exported.")


if __name__ == "__main__":
    main()