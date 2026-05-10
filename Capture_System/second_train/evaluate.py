"""
evaluate.py — 模型效果评估，输出 R² / RMSE 图像

用法：
  python evaluate.py                          # 自动用 checkpoints/best.pt + data/
  python evaluate.py --ckpt checkpoints/best.pt --split test
  python evaluate.py --split all              # 评估全部样本
  python evaluate.py --out results/           # 指定图片输出目录

输出（保存到 --out 目录）：
  scatter.png    — 散点图：预测 vs 真值，标注 R² 和 RMSE
  timeseries.png — 每个样本的时序对比曲线
  metrics.txt    — 数值指标汇总
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def r2_score(truth: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import (UltrasoundDataset, find_all_samples,
                          split_samples, compute_torque_stats, load_sample)
from model import UltraTCN

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = Path(__file__).parent / "checkpoints" / "best.pt"
DEFAULT_OUT  = Path(__file__).parent / "results"


# ── 模型加载 ──────────────────────────────────────────────────────────────────

def load_model_ckpt(ckpt_path: Path):
    ck  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ck["args"]
    m   = UltraTCN(
        us_dim     = cfg.get("us_dim",   32),
        tcn_hidden = cfg.get("hidden",   64),
        tcn_layers = cfg.get("layers",   4),
        dropout    = 0.0,
    ).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, float(ck["t_mean"]), float(ck["t_std"]), int(cfg.get("window", 50))


# ── 对单个 CSV 做流式推理，返回 (pred_arr, truth_arr) ────────────────────────

@torch.no_grad()
def infer_csv(model: UltraTCN, csv_path: Path,
              t_mean: float, t_std: float, W: int):
    us_proc, truth, _ = load_sample(csv_path)
    N = len(truth)

    preds  = np.full(N, np.nan, dtype=np.float32)
    window = []

    for i in range(N):
        window.append(us_proc[i])
        if len(window) > W:
            window.pop(0)
        if len(window) == W:
            x = torch.from_numpy(
                np.stack(window)[:, None, :]   # (W, 1, 300)
            ).unsqueeze(0).to(DEVICE)          # (1, W, 1, 300)
            pred_norm = model(x).item()
            preds[i]  = pred_norm * t_std + t_mean

    # 只保留有预测的帧（去掉前 W-1 帧的 nan）
    valid = ~np.isnan(preds)
    return preds[valid], truth[valid]


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_scatter(all_pred: np.ndarray, all_truth: np.ndarray,
                 r2: float, rmse: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    ax.scatter(all_truth, all_pred, s=2, alpha=0.25, color="#4FC3F7", label="样本点")

    # 45° 理想线
    lo = min(all_truth.min(), all_pred.min())
    hi = max(all_truth.max(), all_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="理想 y=x")

    # 线性回归拟合线
    coef = np.polyfit(all_truth, all_pred, 1)
    xs   = np.linspace(lo, hi, 200)
    ax.plot(xs, np.polyval(coef, xs), color="#EF5350", linewidth=2,
            label=f"拟合线  (slope={coef[0]:.3f})")

    ax.set_xlabel("Ground Truth (Nm)", fontsize=13)
    ax.set_ylabel("Prediction (Nm)",   fontsize=13)
    ax.set_title(
        f"预测 vs 真值\n$R^2$ = {r2:.4f}     RMSE = {rmse:.4f} Nm",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  散点图 → {out_path}")


def plot_timeseries(results: list[tuple[Path, np.ndarray, np.ndarray]],
                    out_path: Path, max_samples: int = 9):
    n  = min(len(results), max_samples)
    nc = 3
    nr = (n + nc - 1) // nc

    fig = plt.figure(figsize=(nc * 5, nr * 3), dpi=150)
    gs  = gridspec.GridSpec(nr, nc, hspace=0.55, wspace=0.35)

    for idx, (csv_path, pred, truth) in enumerate(results[:n]):
        ax  = fig.add_subplot(gs[idx // nc, idx % nc])
        x   = np.arange(len(truth))
        r2  = r2_score(truth, pred)
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))

        ax.plot(x, truth, color="#4FC3F7", linewidth=1.2, label="Truth")
        ax.plot(x, pred,  color="#EF5350", linewidth=1.2, label="Pred",  alpha=0.85)
        ax.set_title(f"{csv_path.stem}\n$R^2$={r2:.3f}  RMSE={rmse:.3f} Nm",
                     fontsize=9)
        ax.set_xlabel("帧", fontsize=8)
        ax.set_ylabel("Torque (Nm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    # 隐藏多余子图
    for idx in range(n, nr * nc):
        fig.add_subplot(gs[idx // nc, idx % nc]).set_visible(False)

    fig.suptitle("各样本时序对比（红=预测，蓝=真值）", fontsize=13, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  时序图 → {out_path}")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def evaluate(args):
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] 权重文件不存在: {ckpt_path}"); sys.exit(1)

    all_paths = find_all_samples()
    if not all_paths:
        print("[ERROR] second_train/data/ 中没有 CSV，请先用 gait_labeler 标注数据")
        sys.exit(1)

    print(f"权重: {ckpt_path}")
    print(f"设备: {DEVICE}")

    model, t_mean, t_std, W = load_model_ckpt(ckpt_path)
    print(f"模型加载完成  窗口 W={W}  t_mean={t_mean:.4f}  t_std={t_std:.4f}")

    # 选取评估集
    if args.split == "all":
        eval_paths = all_paths
        print(f"评估模式: 全部  ({len(eval_paths)} 个样本)")
    else:
        train_p, val_p, test_p = split_samples(
            all_paths,
            train_ratio = args.train_ratio,
            val_ratio   = args.val_ratio,
        )
        if args.split == "test":
            eval_paths = test_p
        elif args.split == "val":
            eval_paths = val_p
        else:
            eval_paths = train_p
        print(f"评估模式: {args.split}  ({len(eval_paths)} 个样本)")

    if not eval_paths:
        print("[WARN] 该划分下样本数为 0，改用全部样本")
        eval_paths = all_paths

    # 逐文件推理
    results: list[tuple[Path, np.ndarray, np.ndarray]] = []
    all_pred, all_truth = [], []

    for csv_path in eval_paths:
        print(f"  推理: {csv_path.name} ...", end=" ", flush=True)
        pred, truth = infer_csv(model, csv_path, t_mean, t_std, W)
        r2   = r2_score(truth, pred)
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        print(f"R²={r2:.4f}  RMSE={rmse:.4f} Nm")
        results.append((csv_path, pred, truth))
        all_pred.append(pred)
        all_truth.append(truth)

    all_pred  = np.concatenate(all_pred)
    all_truth = np.concatenate(all_truth)

    # 整体指标
    r2_total   = r2_score(all_truth, all_pred)
    rmse_total = float(np.sqrt(np.mean((all_pred - all_truth) ** 2)))
    mae_total  = float(np.mean(np.abs(all_pred - all_truth)))

    print(f"\n{'='*50}")
    print(f"  样本数 : {len(eval_paths)}")
    print(f"  总帧数 : {len(all_truth)}")
    print(f"  R²     : {r2_total:.6f}")
    print(f"  RMSE   : {rmse_total:.6f} Nm")
    print(f"  MAE    : {mae_total:.6f} Nm")
    print(f"{'='*50}")

    # 保存图片
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter(all_pred, all_truth, r2_total, rmse_total,
                 out_dir / f"scatter_{args.split}.png")
    plot_timeseries(results, out_dir / f"timeseries_{args.split}.png",
                    max_samples=args.max_ts)

    # 保存数值指标
    metrics_path = out_dir / f"metrics_{args.split}.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint : {ckpt_path}\n")
        f.write(f"split      : {args.split}\n")
        f.write(f"samples    : {len(eval_paths)}\n")
        f.write(f"frames     : {len(all_truth)}\n")
        f.write(f"R2         : {r2_total:.6f}\n")
        f.write(f"RMSE (Nm)  : {rmse_total:.6f}\n")
        f.write(f"MAE  (Nm)  : {mae_total:.6f}\n\n")
        f.write("per-sample:\n")
        for csv_path, pred, truth in results:
            r2   = r2_score(truth, pred)
            rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
            mae  = float(np.mean(np.abs(pred - truth)))
            f.write(f"  {csv_path.name:<35}  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}\n")

    print(f"  指标文件 → {metrics_path}")
    print(f"\n所有结果已保存到: {out_dir.resolve()}")


def main():
    p = argparse.ArgumentParser(description="UltraTCN 评估脚本")
    p.add_argument("--ckpt",        default=str(DEFAULT_CKPT), help="权重文件")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test", "all"], help="评估集划分")
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--out",         default=str(DEFAULT_OUT),  help="图片输出目录")
    p.add_argument("--max_ts",      type=int,   default=9,
                   help="时序图最多显示几个样本（默认 9）")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
