"""
visualize.py — 流式推理可视化（单通道超声 + 叠加曲线）

超声瀑布图上叠加：
  - 红色：Prediction (Nm/kg)
  - 蓝色：Ground Truth (Nm/kg)
  - 绿色：acc_z (m/s²)

用法：
    python visualize.py --seg <handle_data_XX> --ckpt best.pt
    python visualize.py --seg ... --fps 30 --channel 1
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, median_filter, laplace
from scipy.signal import hilbert

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                                   QHBoxLayout, QLabel, QComboBox)
except ImportError:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QComboBox)

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="k", foreground="w")

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import _load_segment, KIN_DIM
from model import TorqueTCN


# ── 超声预处理 ────────────────────────────────────────────────────────────────

def preprocess_us(raw: np.ndarray, tgc=0.025, sigma=1.2, sharpen=0.2) -> np.ndarray:
    """raw: (N, D) → (N, D) float32 [0,1]"""
    rf  = raw.astype(np.float32)
    rf -= rf.mean(axis=1, keepdims=True)
    env = np.abs(hilbert(rf, axis=1))
    log = 20.0 * np.log10(np.clip(env, 1e-6, None))
    out  = np.zeros_like(log)
    gain = np.arange(log.shape[1]) * tgc
    for i in range(len(log)):
        f    = log[i] + gain
        vmin = np.percentile(f, 25)
        vmax = np.percentile(f, 99)
        out[i] = np.clip((f - vmin) / (vmax - vmin + 1e-5), 0, 1)
    out = median_filter(out, size=(3, 3))
    out = gaussian_filter(out, sigma=[0.5, sigma])
    if sharpen > 0:
        out = np.clip(out - sharpen * laplace(out), 0, 1)
    return out.astype(np.float32)


def gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


# ── checkpoint 加载 ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, device):
    ck  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["args"]
    model = TorqueTCN(
        kin_dim        = KIN_DIM,
        us_dim         = cfg.get("us_dim",         32),
        tcn_hidden     = cfg.get("tcn_hidden",     64),
        tcn_layers     = cfg.get("tcn_layers",     3),
        dropout        = 0.0,
        use_ultrasound = not cfg.get("no_ultrasound", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return (model,
            ck["kin_mean"], ck["kin_std"],
            float(ck["lbl_mean"]), float(ck["lbl_std"]),
            cfg.get("window", 50))


# ── 主窗口 ────────────────────────────────────────────────────────────────────

class InferViewer(QWidget):
    HISTORY = 400
    S0, S1  = 150, 850   # 深度显示范围

    def __init__(self, seg_dir, model, kin, kin_n, us_proc_all,
                 lbl, lbl_mean, lbl_std, W, device, fps, channel):
        super().__init__()
        self.setWindowTitle(f"实时推理 — {seg_dir.name}")
        self.resize(1100, 700)

        self.model       = model
        self.kin         = kin          # (N,18) 原始
        self.kin_n       = kin_n        # (N,18) 归一化
        self.us_proc_all = us_proc_all  # dict {ch: (N, D_full)}
        self.lbl         = lbl
        self.lbl_mean    = lbl_mean
        self.lbl_std     = lbl_std
        self.W           = W
        self.device      = device
        self.N           = len(lbl)
        self.channel     = channel
        self.cur         = 0

        # 流式缓冲
        self.kin_buf = deque(maxlen=W)
        self.us_buf  = deque(maxlen=W)   # 存 (4,700) 裁剪后的帧

        # 滚动显示缓冲
        H, D = self.HISTORY, self.S1 - self.S0
        self.us_buf_disp = np.zeros((H, D), dtype=np.float32)
        self.truth_buf   = np.zeros(H, dtype=np.float32)
        self.pred_buf    = np.full(H, np.nan, dtype=np.float32)
        self.acc_buf     = np.zeros(H, dtype=np.float32)

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(max(1, int(1000 / fps)))

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # 顶部工具栏
        top = QHBoxLayout()
        top.addWidget(QLabel("通道:"))
        self.combo_ch = QComboBox()
        for ch in sorted(self.us_proc_all.keys()):
            self.combo_ch.addItem(f"Channel {ch}", ch)
        self.combo_ch.setCurrentIndex(self.channel - 1)
        self.combo_ch.currentIndexChanged.connect(self._on_channel_changed)
        top.addWidget(self.combo_ch)
        top.addStretch()
        self.lbl_stat = QLabel("推理中...")
        self.lbl_stat.setStyleSheet("color:#aaa; font-size:11px")
        top.addWidget(self.lbl_stat)
        root.addLayout(top)

        # 超声瀑布图
        self.pw = pg.PlotWidget(title=f"Channel {self.channel}  —  超声 + 推理叠加")
        self.pw.invertY(True)
        self.pw.setLabel("left",   "Depth (sample)")
        self.pw.setLabel("bottom", "Frame (rolling)")

        self.img_item = pg.ImageItem()
        self.img_item.setLookupTable(gray_lut())
        self.pw.addItem(self.img_item)

        # 叠加曲线：y 值映射到深度范围 [S0, S1]
        self.curve_truth = self.pw.plot(pen=pg.mkPen("#4FC3F7", width=2))
        self.curve_pred  = self.pw.plot(pen=pg.mkPen("#EF5350", width=2))
        self.curve_acc   = self.pw.plot(pen=pg.mkPen("#66BB6A", width=1.5))
        self.curve_truth.setZValue(10)
        self.curve_pred.setZValue(11)
        self.curve_acc.setZValue(9)

        # 图例（用 TextItem 手动标注，避免遮挡）
        for text, color, y_off in [
            ("Ground Truth", "#4FC3F7", 0),
            ("Prediction",   "#EF5350", 20),
            ("acc_z",        "#66BB6A", 40),
        ]:
            t = pg.TextItem(text, color=color, anchor=(0, 0))
            t.setPos(5, self.S0 + y_off)
            t.setZValue(20)
            self.pw.addItem(t)

        root.addWidget(self.pw, stretch=1)

    def _on_channel_changed(self, idx):
        self.channel = self.combo_ch.itemData(idx)
        self.pw.setTitle(f"Channel {self.channel}  —  超声 + 推理叠加")
        # 清空超声显示缓冲，切换通道后重新填充
        self.us_buf_disp[:] = 0

    def _normalize_to_depth(self, arr: np.ndarray, vmin, vmax) -> np.ndarray:
        """把信号值线性映射到 [S0, S1] 深度范围内显示。"""
        span = vmax - vmin if vmax != vmin else 1.0
        return self.S0 + (arr - vmin) / span * (self.S1 - self.S0)

    def _step(self):
        if self.cur >= self.N:
            self.timer.stop()
            return

        i = self.cur

        # ── 推理 ─────────────────────────────────────────────────────────────
        self.kin_buf.append(self.kin_n[i])

        # us_buf 存裁剪后的 (4,700) 帧（与训练一致）
        us_frame_crop = self.us_proc_all[1][i, self.S0:self.S1]   # 先用 ch1 占位
        # 实际推理需要 4 通道，从原始 us_raw 取
        us4 = np.stack([
            self.us_proc_all[ch][i, self.S0:self.S1]
            for ch in sorted(self.us_proc_all)
        ], axis=0)   # (4, 700)
        self.us_buf.append(us4)

        pred_val = np.nan
        if len(self.kin_buf) == self.W:
            k = torch.from_numpy(np.stack(self.kin_buf)).unsqueeze(0).to(self.device)
            u = torch.from_numpy(np.stack(self.us_buf)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_val = self.model(k, u).item() * self.lbl_std + self.lbl_mean

        # ── 滚动缓冲 ──────────────────────────────────────────────────────────
        self.truth_buf   = np.roll(self.truth_buf,   -1); self.truth_buf[-1]   = self.lbl[i]
        self.pred_buf    = np.roll(self.pred_buf,    -1); self.pred_buf[-1]    = pred_val
        self.acc_buf     = np.roll(self.acc_buf,     -1); self.acc_buf[-1]     = self.kin[i, 2]
        self.us_buf_disp = np.roll(self.us_buf_disp, -1, axis=0)
        self.us_buf_disp[-1] = self.us_proc_all[self.channel][i, self.S0:self.S1]

        # ── 绘图 ──────────────────────────────────────────────────────────────
        img_data = (self.us_buf_disp * 255).astype(np.uint8)
        self.img_item.setImage(img_data, autoLevels=False, levels=(0, 255))
        self.img_item.setRect(pg.QtCore.QRectF(0, self.S0, self.HISTORY, self.S1 - self.S0))

        x = np.arange(self.HISTORY)

        # 把力矩/acc_z 映射到深度范围叠加显示
        t_mapped = self._normalize_to_depth(self.truth_buf,
                                            self.truth_buf.min(), self.truth_buf.max())
        self.curve_truth.setData(x, t_mapped)

        valid = ~np.isnan(self.pred_buf)
        if valid.any():
            p_mapped = self._normalize_to_depth(self.pred_buf[valid],
                                                self.truth_buf.min(), self.truth_buf.max())
            self.curve_pred.setData(x[valid], p_mapped)

        a_mapped = self._normalize_to_depth(self.acc_buf,
                                            self.acc_buf.min(), self.acc_buf.max())
        self.curve_acc.setData(x, a_mapped)

        # 状态栏
        if valid.any():
            err  = self.pred_buf[valid] - self.truth_buf[valid]
            mae  = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            self.lbl_stat.setText(
                f"帧 {i+1}/{self.N}   MAE={mae:.4f}   RMSE={rmse:.4f} Nm/kg")

        self.cur += 1


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seg",     required=True)
    p.add_argument("--ckpt",    default="checkpoints/best.pt")
    p.add_argument("--fps",     type=float, default=30.0)
    p.add_argument("--channel", type=int,   default=1, help="默认显示通道 1~4")
    args = p.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_dir   = Path(args.seg)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent / ckpt_path

    for path, name in [(seg_dir, "段目录"), (ckpt_path, "权重文件")]:
        if not path.exists():
            print(f"[ERROR] {name}不存在: {path}"); sys.exit(1)

    print("加载模型...")
    model, kin_mean, kin_std, lbl_mean, lbl_std, W = load_checkpoint(ckpt_path, device)

    print("加载运动学 + 标签...")
    kin, _, lbl = _load_segment(seg_dir)
    kin_n = (kin - kin_mean) / kin_std

    print("加载并预处理超声（4通道）...")
    us_df  = pd.read_csv(seg_dir / "input" / "ultrasound.csv")
    dcols  = [c for c in us_df.columns if c.startswith("d") and c[1:].isdigit()]
    us_proc_all = {}
    for ch in sorted(us_df["channel"].unique()):
        sub = us_df[us_df["channel"] == ch].reset_index(drop=True)
        raw = sub[dcols].values.astype(np.float32)
        print(f"  Channel {int(ch)}...")
        us_proc_all[int(ch)] = preprocess_us(raw)

    print(f"总帧数: {len(lbl)}  窗口: {W}  fps: {args.fps}")

    app = QApplication(sys.argv)
    win = InferViewer(seg_dir, model, kin, kin_n, us_proc_all,
                      lbl, lbl_mean, lbl_std, W, device, args.fps, args.channel)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
