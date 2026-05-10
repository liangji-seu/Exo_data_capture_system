"""
train.py — 双通道中性参考 CNN-GRU 训练脚本

用法：
    python train.py --data_root 超声采集数据/第二批试采用（慢，中速行走）
    python train.py --data_root ... --no_ultrasound
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))              # LSTM/ 优先，保证 model/dataset 用本目录的
sys.path.insert(1, str(_HERE.parent))       # tcn_train/ 次之，用于 data_loader
from data_loader import (_find_all_segments, split_segments,
                         compute_kin_stats, compute_label_stats, KIN_DIM)
from dataset import DualChannelDataset
from model import TorqueCNNGRU


def criterion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse  = F.mse_loss(pred, target)
    vp   = pred   - pred.mean()
    vt   = target - target.mean()
    corr = (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-8)
    return mse + 1.0 * (1 - corr)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for kin, us, lbl in loader:
            kin, us, lbl = kin.to(device), us.to(device), lbl.to(device)
            pred = model(kin, us)
            loss = criterion(pred, lbl)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).parent.parent.parent / data_root
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    all_segs = _find_all_segments(data_root)
    print(f"共 {len(all_segs)} 段数据")
    train_segs, val_segs, _ = split_segments(all_segs)
    print(f"train={len(train_segs)}, val={len(val_segs)}")

    stats_path = Path(__file__).parent / "kin_stats.npz"
    if stats_path.exists():
        d = np.load(stats_path)
        kin_mean, kin_std = d["mean"], d["std"]
        lbl_mean, lbl_std = float(d["lbl_mean"]), float(d["lbl_std"])
        print("加载已有统计量")
    else:
        print("计算统计量...")
        kin_mean, kin_std = compute_kin_stats(train_segs)
        lbl_mean, lbl_std = compute_label_stats(train_segs)
        np.savez(stats_path, mean=kin_mean, std=kin_std,
                 lbl_mean=lbl_mean, lbl_std=lbl_std)
    print(f"标签: mean={lbl_mean:.3f} Nm, std={lbl_std:.3f} Nm")

    W = args.window
    train_ds = DualChannelDataset(train_segs, W, stride=args.stride,
                                  kin_mean=kin_mean, kin_std=kin_std,
                                  lbl_mean=lbl_mean, lbl_std=lbl_std,
                                  ref_frames=args.ref_frames, augment=True)
    val_ds   = DualChannelDataset(val_segs,   W, stride=args.stride,
                                  kin_mean=kin_mean, kin_std=kin_std,
                                  lbl_mean=lbl_mean, lbl_std=lbl_std,
                                  ref_frames=args.ref_frames, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    model = TorqueCNNGRU(
        kin_dim        = KIN_DIM,
        us_feat        = args.us_feat,
        gru_hidden     = args.gru_hidden,
        gru_layers     = args.gru_layers,
        dropout        = args.dropout,
        use_ultrasound = not args.no_ultrasound,
    ).to(device)
    if args.no_ultrasound:
        print("超声分支已屏蔽")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    save_dir = Path(__file__).parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    best_val, no_improve = float("inf"), 0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step(va)
        history["train"].append(tr); history["val"].append(va)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val={va:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s")

        if va < best_val:
            best_val, no_improve = va, 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_loss": va,
                "kin_mean": kin_mean, "kin_std": kin_std,
                "lbl_mean": lbl_mean, "lbl_std": lbl_std,
                "args": vars(args),
            }, save_dir / "best.pt")
            print(f"  → 保存最优 (val={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"早停：{args.patience} 轮无改善"); break

    np.savez(save_dir / "history.npz",
             train=np.array(history["train"]),
             val=np.array(history["val"]))
    print(f"\n训练完成，最优 val_loss={best_val:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--window",      type=int,   default=50)
    p.add_argument("--stride",      type=int,   default=1)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--us_feat",     type=int,   default=64)
    p.add_argument("--gru_hidden",  type=int,   default=128)
    p.add_argument("--gru_layers",  type=int,   default=2)
    p.add_argument("--ref_frames",  type=int,   default=30)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--no_ultrasound", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
