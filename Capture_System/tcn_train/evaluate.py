"""
evaluate.py — 测试集评估脚本

用法：
    python evaluate.py --data_root ../超声采集数据/第二批试采用（慢，中速行走）
                       --ckpt checkpoints/best.pt
                       --plot   # 可选，画预测 vs 真实曲线
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_loader import (GaitDataset, _find_all_segments,
                         split_segments, KIN_DIM)
from model import TorqueTCN
from train import TorchGaitDataset


# ── 指标 ──────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ── 推理 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for kin, us, lbl in loader:
        out = model(kin.to(device), us.to(device))
        preds.append(out.cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(labels).squeeze()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})
    kin_mean  = ckpt["kin_mean"]
    kin_std   = ckpt["kin_std"]
    lbl_mean  = float(ckpt.get("lbl_mean", 0.0))
    lbl_std   = float(ckpt.get("lbl_std",  1.0))
    W         = ckpt_args.get("window", 50)

    model = TorqueTCN(
        kin_dim        = KIN_DIM,
        us_dim         = ckpt_args.get("us_dim",         32),
        tcn_hidden     = ckpt_args.get("tcn_hidden",     64),
        tcn_layers     = ckpt_args.get("tcn_layers",     3),
        dropout        = 0.0,
        use_ultrasound = not ckpt_args.get("no_ultrasound", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"加载模型: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")
    print(f"标签统计: mean={lbl_mean:.3f}, std={lbl_std:.3f}")

    # 数据
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).parent.parent / data_root
    all_segs  = _find_all_segments(data_root)
    _, _, test_segs = split_segments(all_segs)
    print(f"测试段数: {len(test_segs)}")

    test_ds     = TorchGaitDataset(GaitDataset(test_segs, W, stride=1,
                                               kin_mean=kin_mean, kin_std=kin_std,
                                               lbl_mean=lbl_mean, lbl_std=lbl_std))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"测试样本数: {len(test_ds)}")

    # 推理（输出是归一化空间，反归一化回 Nm）
    y_pred_norm, y_true_norm = predict(model, test_loader, device)
    y_pred = y_pred_norm * lbl_std + lbl_mean
    y_true = y_true_norm * lbl_std + lbl_mean

    # 指标
    print("\n── 测试集指标 ──────────────────────────────")
    print(f"  RMSE : {rmse(y_true, y_pred):.4f} Nm/kg")
    print(f"  MAE  : {mae(y_true,  y_pred):.4f} Nm/kg")
    print(f"  R²   : {r2(y_true,   y_pred):.4f}")
    print(f"  真实值范围: [{y_true.min():.3f}, {y_true.max():.3f}] Nm/kg")
    print(f"  预测值范围: [{y_pred.min():.3f}, {y_pred.max():.3f}] Nm/kg")

    # 保存预测结果
    out_path = Path(args.ckpt).parent / "test_predictions.npz"
    np.savez(out_path, y_pred=y_pred, y_true=y_true)
    print(f"\n预测结果已保存: {out_path}")

    # 可选：画图
    if args.plot:
        _plot(y_true, y_pred, Path(args.ckpt).parent)


def _plot(y_true, y_pred, save_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 时序对比（取前 500 帧）
    n = min(500, len(y_true))
    axes[0].plot(y_true[:n],  label="真实力矩", linewidth=1.2)
    axes[0].plot(y_pred[:n],  label="预测力矩", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("帧"); axes[0].set_ylabel("力矩 (Nm)")
    axes[0].set_title("预测 vs 真实（前500帧）")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 散点图
    axes[1].scatter(y_true, y_pred, s=2, alpha=0.3)
    lim = [min(y_true.min(), y_pred.min()) - 1,
           max(y_true.max(), y_pred.max()) + 1]
    axes[1].plot(lim, lim, "r--", linewidth=1)
    axes[1].set_xlabel("真实力矩 (Nm)"); axes[1].set_ylabel("预测力矩 (Nm)")
    axes[1].set_title(f"散点图  R²={r2(y_true, y_pred):.3f}  RMSE={rmse(y_true, y_pred):.3f} Nm")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = save_dir / "evaluation.png"
    plt.savefig(fig_path, dpi=150)
    print(f"图表已保存: {fig_path}")
    plt.show()


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt",      default="checkpoints/best.pt")
    p.add_argument("--plot",      action="store_true")
    evaluate(p.parse_args())
