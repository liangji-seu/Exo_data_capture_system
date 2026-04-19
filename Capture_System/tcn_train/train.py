"""
train.py — 训练脚本

用法：
    python train.py --data_root ../超声采集数据/第二批试采用（慢，中速行走）
                    --epochs 100 --batch 16 --window 50 --lr 1e-3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_loader import (GaitDataset, _find_all_segments,
                         split_segments, compute_kin_stats,
                         compute_label_stats, KIN_DIM)
from model import TorqueTCN


# ── PyTorch Dataset 包装 ──────────────────────────────────────────────────────

class TorchGaitDataset(Dataset):
    def __init__(self, gait_ds: GaitDataset):
        self._ds = gait_ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        kin, us, lbl = self._ds[idx]
        return (torch.from_numpy(kin),
                torch.from_numpy(us),
                torch.from_numpy(lbl))


# ── 训练 / 验证单轮 ───────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for kin, us, lbl in loader:
            kin = kin.to(device)
            us  = us.to(device)
            lbl = lbl.to(device)

            pred = model(kin, us)          # (B, 1)
            loss = criterion(pred, lbl)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── 主训练流程 ────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据（支持相对路径，基准为本脚本的上级目录 Capture_System/）
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).parent.parent / data_root
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    all_segs  = _find_all_segments(data_root)
    print(f"数据目录: {data_root}")
    print(f"共 {len(all_segs)} 个数据段")

    train_segs, val_segs, test_segs = split_segments(all_segs)
    print(f"划分: train={len(train_segs)}, val={len(val_segs)}, test={len(test_segs)}")

    # 统计量（仅在训练集上计算）
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
    print(f"标签统计: mean={lbl_mean:.3f} Nm/kg, std={lbl_std:.3f} Nm/kg")

    W = args.window
    train_ds = TorchGaitDataset(GaitDataset(train_segs, W, stride=args.stride,
                                            kin_mean=kin_mean, kin_std=kin_std,
                                            lbl_mean=lbl_mean, lbl_std=lbl_std))
    val_ds   = TorchGaitDataset(GaitDataset(val_segs,   W, stride=args.stride,
                                            kin_mean=kin_mean, kin_std=kin_std,
                                            lbl_mean=lbl_mean, lbl_std=lbl_std))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    # 模型
    model = TorqueTCN(
        kin_dim        = KIN_DIM,
        us_dim         = args.us_dim,
        tcn_hidden     = args.tcn_hidden,
        tcn_layers     = args.tcn_layers,
        dropout        = args.dropout,
        use_ultrasound = not args.no_ultrasound,
    ).to(device)
    if args.no_ultrasound:
        print("超声分支已屏蔽，仅使用 IMU + 电机数据")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    def criterion(pred, target):
        mse  = F.mse_loss(pred, target)
        # 相关系数损失：惩罚幅值压缩，鼓励预测跟上峰值
        vp   = pred   - pred.mean()
        vt   = target - target.mean()
        corr = (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-8)
        return mse + 1.0 * (1 - corr)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    # 训练循环
    save_dir = Path(__file__).parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)
    best_val  = float("inf")
    no_improve = 0
    history   = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(va_loss)

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_loss:.4f}  val={va_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  {elapsed:.1f}s")

        # 保存最优模型
        if va_loss < best_val:
            best_val   = va_loss
            no_improve = 0
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "val_loss": va_loss,
                "kin_mean": kin_mean,
                "kin_std":  kin_std,
                "lbl_mean": lbl_mean,
                "lbl_std":  lbl_std,
                "args":     vars(args),
            }, save_dir / "best.pt")
            print(f"  → 保存最优模型 (val={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"早停：{args.patience} 轮无改善")
                break

    # 保存训练曲线
    np.savez(save_dir / "history.npz",
             train=np.array(history["train"]),
             val=np.array(history["val"]))
    print(f"\n训练完成，最优 val_loss={best_val:.4f}")
    print(f"模型保存在: {save_dir / 'best.pt'}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="第二批试采用目录路径")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--window",     type=int,   default=50,
                   help="滑动窗口帧数 W")
    p.add_argument("--stride",     type=int,   default=5,
                   help="滑动窗口步长，越大样本重叠越少")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--us_dim",     type=int,   default=32,
                   help="超声编码器输出维度")
    p.add_argument("--tcn_hidden", type=int,   default=64)
    p.add_argument("--tcn_layers", type=int,   default=3)
    p.add_argument("--patience",   type=int,   default=20,
                   help="早停耐心轮数")
    p.add_argument("--no_ultrasound", action="store_true",
                   help="屏蔽超声分支，仅用 IMU + 电机数据")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
