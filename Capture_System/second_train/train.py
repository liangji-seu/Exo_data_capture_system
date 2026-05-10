"""
train.py — 单通道超声 TCN 训练脚本

用法：
  python train.py
  python train.py --epochs 200 --window 60 --batch 64 --lr 5e-4

训练完成后在 checkpoints/ 下生成：
  best.pt    — 最低验证损失的权重
  last.pt    — 最终 epoch 权重
  stats.json — 力矩归一化统计量（推理时需要）
  history.pt — 每 epoch 的 train/val 损失列表
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 同目录导入
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import UltrasoundDataset, find_all_samples, split_samples, compute_torque_stats
from model import UltraTCN

CKPT_DIR = Path(__file__).parent / "checkpoints"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, t_std: float) -> tuple[float, float]:
    """返回 (loss, MAE_Nm)。MAE 已反归一化到原始力矩量纲。"""
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for us, lbl in loader:
            us, lbl = us.to(DEVICE), lbl.to(DEVICE)
            pred = model(us)
            total_loss += criterion(pred, lbl).item() * len(us)
            total_mae  += (pred - lbl).abs().sum().item() * t_std
            n          += len(us)
    return total_loss / n, total_mae / n


def train(args: argparse.Namespace):
    all_paths = find_all_samples()
    if not all_paths:
        raise RuntimeError("second_train/data/ 中没有 *.csv 文件，请先用 gait_labeler 标注数据")

    train_p, val_p, test_p = split_samples(all_paths, args.train_ratio, args.val_ratio)
    print(f"样本总数: {len(all_paths)}  训练: {len(train_p)}  验证: {len(val_p)}  测试: {len(test_p)}")
    print(f"使用设备: {DEVICE}")

    t_mean, t_std = compute_torque_stats(train_p)
    print(f"力矩统计 (训练集): mean={t_mean:.4f}  std={t_std:.4f}")

    train_ds = UltrasoundDataset(train_p, args.window, args.stride, t_mean, t_std)
    val_ds   = UltrasoundDataset(val_p,   args.window, 1,           t_mean, t_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=(DEVICE == "cuda"))

    model = UltraTCN(
        us_dim     = args.us_dim,
        tcn_hidden = args.hidden,
        tcn_layers = args.layers,
        dropout    = args.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    CKPT_DIR.mkdir(exist_ok=True)
    best_val_loss  = float("inf")
    history        = []
    start_epoch    = 1
    no_improve     = 0   # early stopping 计数器

    # ── 断点续训 ─────────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"断点文件不存在: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history       = ckpt.get("history", [])
        print(f"从断点恢复: {ckpt_path}  (epoch {ckpt['epoch']}，best_val={best_val_loss:.5f})")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── 训练 ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for us, lbl in train_loader:
            us, lbl = us.to(DEVICE), lbl.to(DEVICE)
            pred = model(us)
            loss = criterion(pred, lbl)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(us)
        train_loss /= len(train_ds)

        # ── 验证 ─────────────────────────────────────────────────────────
        val_loss, val_mae = evaluate(model, val_loader, criterion, t_std)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_loss,
                         "val": val_loss, "val_mae_Nm": val_mae})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"MAE={val_mae:.4f} Nm  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({
                "epoch":          epoch,
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "best_val_loss":  best_val_loss,
                "history":        history,
                "t_mean":         t_mean,
                "t_std":          t_std,
                "args":           vars(args),
                "val_mae":        val_mae,
            }, CKPT_DIR / "best.pt")
            print(f"  ✓ 最佳模型已保存 (epoch {epoch}, val={val_loss:.5f}, MAE={val_mae:.4f} Nm)")

        # 定期断点保存
        if epoch % args.save_every == 0:
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "history":       history,
                "t_mean":        t_mean,
                "t_std":         t_std,
                "args":          vars(args),
            }, CKPT_DIR / f"epoch_{epoch:04d}.pt")
            print(f"  → 断点已保存: epoch_{epoch:04d}.pt")

        # early stopping（独立判断，与断点保存无关）
        if val_loss >= best_val_loss:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"\nEarly stopping：验证集 {args.patience} 轮无改善，停止训练 (epoch {epoch})")
                break

    # ── 最终保存 ─────────────────────────────────────────────────────────
    torch.save({
        "epoch":         args.epochs,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "history":       history,
        "t_mean":        t_mean,
        "t_std":         t_std,
        "args":          vars(args),
    }, CKPT_DIR / "last.pt")

    with open(CKPT_DIR / "stats.json", "w") as f:
        json.dump({"torque_mean": t_mean, "torque_std": t_std}, f, indent=2)

    torch.save(history, CKPT_DIR / "history.pt")

    print(f"\n训练完成。最佳验证损失: {best_val_loss:.5f}")
    print(f"权重保存: {CKPT_DIR}/best.pt")

    # ── 测试集评估 ───────────────────────────────────────────────────────
    if test_p:
        test_ds     = UltrasoundDataset(test_p, args.window, 1, t_mean, t_std)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        ckpt        = torch.load(CKPT_DIR / "best.pt", map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        test_loss, test_mae = evaluate(model, test_loader, criterion, t_std)
        print(f"测试集  loss={test_loss:.5f}  MAE={test_mae:.4f} Nm")


def main():
    parser = argparse.ArgumentParser(description="单通道超声 TCN 力矩预测训练")
    parser.add_argument("--window",      type=int,   default=50,   help="滑窗大小（帧数）")
    parser.add_argument("--stride",      type=int,   default=5,    help="训练集滑窗步长")
    parser.add_argument("--batch",       type=int,   default=32,   help="批大小")
    parser.add_argument("--epochs",      type=int,   default=150,  help="训练轮数")
    parser.add_argument("--lr",          type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--us_dim",      type=int,   default=32,   help="超声编码器输出维度")
    parser.add_argument("--hidden",      type=int,   default=64,   help="TCN 隐层通道数")
    parser.add_argument("--layers",      type=int,   default=4,    help="TCN 层数")
    parser.add_argument("--dropout",     type=float, default=0.4,   help="Dropout 比例（建议 0.4~0.5）")
    parser.add_argument("--wd",          type=float, default=1e-3,  help="Weight decay")
    parser.add_argument("--patience",    type=int,   default=30,    help="Early stopping 轮数（0=关闭）")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="训练集比例")
    parser.add_argument("--val_ratio",   type=float, default=0.15, help="验证集比例")
    parser.add_argument("--save_every",  type=int,   default=20,   help="每隔 N epoch 保存一次断点")
    parser.add_argument("--resume",      type=str,   default="",   help="从断点文件继续训练，如 checkpoints/epoch_0040.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
