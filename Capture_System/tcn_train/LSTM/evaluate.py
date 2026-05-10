"""
evaluate.py — 双通道中性参考 CNN-GRU 测试集评估

用法：
    python evaluate.py --data_root 超声采集数据/第二批试采用（慢，中速行走）
                       --ckpt checkpoints/best.pt --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from data_loader import _find_all_segments, split_segments, KIN_DIM
from dataset import DualChannelDataset
from model import TorqueCNNGRU


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def r2(a, b):   return float(1 - np.sum((a-b)**2) / (np.sum((a-np.mean(a))**2) + 1e-10))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for kin, us, lbl in loader:
        out = model(kin.to(device), us.to(device))
        preds.append(out.cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(preds).squeeze(), np.concatenate(labels).squeeze()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg      = ckpt.get("args", {})
    kin_mean = ckpt["kin_mean"]
    kin_std  = ckpt["kin_std"]
    lbl_mean = float(ckpt.get("lbl_mean", 0.0))
    lbl_std  = float(ckpt.get("lbl_std",  1.0))
    W        = cfg.get("window", 50)

    model = TorqueCNNGRU(
        kin_dim        = KIN_DIM,
        us_feat        = cfg.get("us_feat",     64),
        gru_hidden     = cfg.get("gru_hidden",  128),
        gru_layers     = cfg.get("gru_layers",  2),
        dropout        = 0.0,
        use_ultrasound = not cfg.get("no_ultrasound", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"加载模型: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).parent.parent.parent / data_root
    all_segs = _find_all_segments(data_root)
    _, _, test_segs = split_segments(all_segs)
    print(f"测试段数: {len(test_segs)}")

    test_ds = DualChannelDataset(
        test_segs, W, stride=1,
        kin_mean=kin_mean, kin_std=kin_std,
        lbl_mean=lbl_mean, lbl_std=lbl_std,
        ref_frames=cfg.get("ref_frames", 30), augment=False)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"测试样本: {len(test_ds)}")

    y_pred_n, y_true_n = predict(model, loader, device)
    y_pred = y_pred_n * lbl_std + lbl_mean
    y_true = y_true_n * lbl_std + lbl_mean

    print("\n── 测试集指标 ──────────────────────────────")
    print(f"  RMSE : {rmse(y_true, y_pred):.4f} Nm")
    print(f"  MAE  : {mae(y_true,  y_pred):.4f} Nm")
    print(f"  R²   : {r2(y_true,   y_pred):.4f}")
    print(f"  真实值范围: [{y_true.min():.3f}, {y_true.max():.3f}] Nm")
    print(f"  预测值范围: [{y_pred.min():.3f}, {y_pred.max():.3f}] Nm")

    out_path = Path(args.ckpt).parent / "test_predictions.npz"
    np.savez(out_path, y_pred=y_pred, y_true=y_true)
    print(f"预测结果已保存: {out_path}")

    if args.plot:
        _plot(y_true, y_pred, Path(args.ckpt).parent)


def _plot(y_true, y_pred, save_dir):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("matplotlib 未安装"); return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    n = min(500, len(y_true))
    axes[0].plot(y_true[:n], label="真实", linewidth=1.2)
    axes[0].plot(y_pred[:n], label="预测", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("帧"); axes[0].set_ylabel("力矩 (Nm)")
    axes[0].set_title("预测 vs 真实（前500帧）")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true, y_pred, s=2, alpha=0.3)
    lim = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    axes[1].plot(lim, lim, "r--")
    axes[1].set_xlabel("真实 (Nm)"); axes[1].set_ylabel("预测 (Nm)")
    axes[1].set_title(f"散点图  R²={r2(y_true,y_pred):.3f}  RMSE={rmse(y_true,y_pred):.3f} Nm")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = save_dir / "evaluation.png"
    plt.savefig(fig_path, dpi=150)
    print(f"图表已保存: {fig_path}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt",      default="checkpoints/best.pt")
    p.add_argument("--plot",      action="store_true")
    evaluate(p.parse_args())
