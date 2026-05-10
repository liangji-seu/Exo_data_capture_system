#!/usr/bin/env python3
"""
infer_compare.py — 流式推理 + 力矩对比可视化

界面布局：
  左   B-mode 超声瀑布图（滚动）
  右上  力矩对比折线图（红=预测，蓝=真值，滚动）
  右下  全程力矩对比（随推理逐帧积累，结束后完整显示）
  顶栏  实时 R² / RMSE

用法：
  python infer_compare.py
  python infer_compare.py --csv data/20260422_141047_1.csv
  python infer_compare.py --csv data/xxx.csv --ckpt checkpoints/best.pt --fps 60
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, laplace, median_filter
from scipy.signal import hilbert

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QDoubleSpinBox,
    )
except ImportError:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QDoubleSpinBox,
    )

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="w", foreground="k")

sys.path.insert(0, str(Path(__file__).parent))
from model import UltraTCN
from data_loader import load_sample

HISTORY      = 300   # 滚动窗口帧数
US_N         = 300   # ROI 采样点
DEFAULT_CKPT = Path(__file__).parent / "checkpoints" / "best.pt"
DEFAULT_DATA = Path(__file__).parent / "data"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _preprocess(raw: np.ndarray,
                tgc_slope=0.025, sigma_depth=1.2,
                sigma_time=0.8,  sharpen=0.2) -> np.ndarray:
    raw_f = raw.astype(np.float32)
    n, ns = raw_f.shape
    rf    = raw_f - raw_f.mean(axis=1, keepdims=True)
    env   = np.abs(hilbert(rf, axis=1))
    log   = 20.0 * np.log10(np.clip(env, 1e-6, None))
    out   = np.zeros_like(log)
    gain  = np.arange(ns) * tgc_slope
    for i in range(n):
        f      = log[i] + gain
        vmin   = np.percentile(f, 25.0)
        vmax   = np.percentile(f, 99.0)
        out[i] = np.clip((f - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)
    out = median_filter(out, size=(3, 3))
    if sigma_depth > 0 or sigma_time > 0:
        out = gaussian_filter(out, sigma=[sigma_time, sigma_depth])
    if sharpen > 0:
        out = np.clip(out - sharpen * laplace(out), 0.0, 1.0)
    return out.astype(np.float32)


def _gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def load_model_ckpt(ckpt_path: Path, device: torch.device):
    ck  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["args"]
    m   = UltraTCN(
        us_dim     = cfg.get("us_dim",   32),
        tcn_hidden = cfg.get("hidden",   64),
        tcn_layers = cfg.get("layers",   4),
        dropout    = 0.0,
    ).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, float(ck["t_mean"]), float(ck["t_std"]), int(cfg.get("window", 50))


# ── 主窗口 ────────────────────────────────────────────────────────────────────

class InferCompare(QWidget):
    def __init__(self, device: torch.device,
                 csv_path: str = "", ckpt_path: str = "", fps: float = 30.0):
        super().__init__()
        self.setWindowTitle("UltraTCN 推理对比")
        self.resize(1400, 860)

        self._device  = device
        self._model   = None
        self._t_mean  = 0.0
        self._t_std   = 1.0
        self._W       = 50

        # 数据
        self._us_proc: np.ndarray = np.array([])
        self._truth:   np.ndarray = np.array([])
        self._N        = 0
        self._cur      = 0
        self._playing  = False

        # 滑窗
        self._us_win: deque = deque()

        # 滚动缓冲（用于左侧 B-mode + 右上折线）
        self._us_disp   = np.zeros((HISTORY, US_N), dtype=np.float32)
        self._pred_roll = np.full(HISTORY, np.nan, dtype=np.float32)
        self._truth_roll= np.zeros(HISTORY, dtype=np.float32)

        # 全程积累（用于右下完整对比图）
        self._pred_all:  list[float] = []
        self._truth_all: list[float] = []

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)

        self._build_ui()

        if ckpt_path and Path(ckpt_path).exists():
            self._load_ckpt(ckpt_path)
        if csv_path and Path(csv_path).exists():
            self._load_csv(csv_path)
        self._set_fps(fps)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── 顶栏 ─────────────────────────────────────────────────────────────
        bar = QHBoxLayout()

        btn_ckpt = QPushButton("📦 加载权重")
        btn_ckpt.clicked.connect(self._on_open_ckpt)
        self._lbl_ckpt = QLabel("未加载权重")
        self._lbl_ckpt.setStyleSheet("color:#f90")

        btn_csv = QPushButton("📂 打开 CSV")
        btn_csv.clicked.connect(self._on_open_csv)
        self._lbl_csv = QLabel("未加载数据")
        self._lbl_csv.setStyleSheet("color:#888")

        self._btn_play = QPushButton("▶ 播放")
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._on_play_pause)

        btn_reset = QPushButton("⏮ 重置")
        btn_reset.clicked.connect(self._reset)

        bar.addWidget(btn_ckpt); bar.addWidget(self._lbl_ckpt)
        bar.addSpacing(12)
        bar.addWidget(btn_csv);  bar.addWidget(self._lbl_csv)
        bar.addSpacing(12)
        bar.addWidget(self._btn_play); bar.addWidget(btn_reset)

        bar.addSpacing(20)
        bar.addWidget(QLabel("FPS:"))
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1, 200); self._spin_fps.setValue(30)
        self._spin_fps.setFixedWidth(70)
        self._spin_fps.valueChanged.connect(self._set_fps)
        bar.addWidget(self._spin_fps)

        bar.addStretch()
        self._lbl_metric = QLabel("R² = —    RMSE = — Nm")
        self._lbl_metric.setStyleSheet(
            "color:#FFD54F; font-size:14px; font-weight:bold; padding:0 8px")
        bar.addWidget(self._lbl_metric)
        root.addLayout(bar)

        # ── 主区域：左=B-mode，右=上下两图 ──────────────────────────────────
        main = QHBoxLayout()
        main.setSpacing(6)

        # 左：B-mode 瀑布
        lut = _gray_lut()
        self._pw_us = pg.PlotWidget(title="超声 B-mode（滚动）")
        self._pw_us.setLabel("bottom", "帧")
        self._pw_us.setLabel("left",   "深度（ROI）")
        self._pw_us.setMinimumWidth(480)

        self._img = pg.ImageItem()
        self._img.setLookupTable(lut)
        self._pw_us.addItem(self._img)

        # 力矩叠加在 B-mode 上
        self._c_truth_us = self._pw_us.plot(pen=pg.mkPen("#1565C0", width=2))
        self._c_pred_us  = self._pw_us.plot(pen=pg.mkPen("#C62828", width=2))
        self._c_truth_us.setZValue(10); self._c_pred_us.setZValue(11)
        for txt, col, y in [("Truth", "#1565C0", 18), ("Pred", "#C62828", 38)]:
            t = pg.TextItem(txt, color=col, anchor=(0, 0))
            t.setPos(4, y); t.setZValue(20)
            self._pw_us.addItem(t)
        main.addWidget(self._pw_us, stretch=5)

        # 右：上下两个折线图
        right = QVBoxLayout()
        right.setSpacing(6)

        # 右上：滚动折线图
        self._pw_roll = pg.PlotWidget(title="力矩对比（滚动窗口）")
        self._pw_roll.setLabel("bottom", "帧（相对）")
        self._pw_roll.setLabel("left",   "Torque (Nm)")
        self._pw_roll.showGrid(x=True, y=True, alpha=0.25)
        self._pw_roll.addLegend(offset=(10, 10))
        self._c_truth_roll = self._pw_roll.plot(
            pen=pg.mkPen("#1565C0", width=2), name="Ground Truth")
        self._c_pred_roll  = self._pw_roll.plot(
            pen=pg.mkPen("#C62828", width=2), name="Prediction")
        right.addWidget(self._pw_roll, stretch=1)

        # 右下：全程积累图
        self._pw_full = pg.PlotWidget(title="全程力矩对比（逐帧积累）")
        self._pw_full.setLabel("bottom", "帧（绝对）")
        self._pw_full.setLabel("left",   "Torque (Nm)")
        self._pw_full.showGrid(x=True, y=True, alpha=0.25)
        self._pw_full.addLegend(offset=(10, 10))
        self._c_truth_full = self._pw_full.plot(
            pen=pg.mkPen("#1565C0", width=1.5), name="Ground Truth")
        self._c_pred_full  = self._pw_full.plot(
            pen=pg.mkPen("#C62828", width=1.5), name="Prediction")
        right.addWidget(self._pw_full, stretch=1)

        main.addLayout(right, stretch=6)
        root.addLayout(main, stretch=1)

        # 底部状态栏
        self._lbl_status = QLabel("就绪 — 请加载权重和 CSV 文件")
        self._lbl_status.setStyleSheet("color:#aaa; font-size:11px")
        root.addWidget(self._lbl_status)

    # ── 文件操作 ──────────────────────────────────────────────────────────────

    def _on_open_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择权重", str(DEFAULT_CKPT.parent), "PyTorch (*.pt *.pth)")
        if p: self._load_ckpt(p)

    def _load_ckpt(self, path: str):
        try:
            self._model, self._t_mean, self._t_std, self._W = \
                load_model_ckpt(Path(path), self._device)
            self._us_win = deque(maxlen=self._W)
            self._lbl_ckpt.setText(f"{Path(path).name}  W={self._W}")
            self._lbl_ckpt.setStyleSheet("color:#4FC3F7")
            self._lbl_status.setText(f"权重已加载: {Path(path).name}")
            self._try_enable_play()
        except Exception as e:
            self._lbl_ckpt.setText(f"失败: {e}")
            self._lbl_ckpt.setStyleSheet("color:#f44")

    def _on_open_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择标注 CSV", str(DEFAULT_DATA), "CSV (*.csv)")
        if p: self._load_csv(p)

    def _load_csv(self, path: str):
        try:
            self._lbl_csv.setText("预处理中…")
            self._lbl_status.setText("正在预处理超声数据，请稍候…")
            QApplication.processEvents()

            us_proc, truth, _ = load_sample(Path(path))
            self._us_proc = us_proc
            self._truth   = truth
            self._N       = len(truth)
            self._reset()
            self._lbl_csv.setText(f"{Path(path).name}  ({self._N} 帧)")
            self._lbl_csv.setStyleSheet("color:#aaa")
            self._lbl_status.setText(
                f"数据已加载: {self._N} 帧  — 点击播放开始推理")
            self._try_enable_play()
        except Exception as e:
            self._lbl_csv.setText(f"失败: {e}")
            self._lbl_csv.setStyleSheet("color:#f44")

    def _try_enable_play(self):
        if self._model is not None and self._N > 0:
            self._btn_play.setEnabled(True)

    # ── 播放控制 ──────────────────────────────────────────────────────────────

    def _on_play_pause(self):
        if self._playing:
            self._timer.stop()
            self._playing = False
            self._btn_play.setText("▶ 播放")
        else:
            if self._cur >= self._N:
                self._reset()
            self._timer.start()
            self._playing = True
            self._btn_play.setText("⏸ 暂停")

    def _reset(self):
        self._timer.stop()
        self._playing = False
        self._btn_play.setText("▶ 播放")
        self._cur       = 0
        self._us_win    = deque(maxlen=self._W)
        self._us_disp   = np.zeros((HISTORY, US_N), dtype=np.float32)
        self._pred_roll = np.full(HISTORY, np.nan, dtype=np.float32)
        self._truth_roll= np.zeros(HISTORY, dtype=np.float32)
        self._pred_all  = []
        self._truth_all = []
        self._img.clear()
        for c in [self._c_truth_us, self._c_pred_us,
                  self._c_truth_roll, self._c_pred_roll,
                  self._c_truth_full, self._c_pred_full]:
            c.setData([], [])
        self._lbl_metric.setText("R² = —    RMSE = — Nm")

    def _set_fps(self, v: float):
        self._timer.setInterval(max(1, int(1000 / v)))

    # ── 逐帧推理 ──────────────────────────────────────────────────────────────

    def _step(self):
        if self._model is None or self._N == 0:
            return
        if self._cur >= self._N:
            self._timer.stop()
            self._playing = False
            self._btn_play.setText("▶ 播放")
            self._on_done()
            return

        i = self._cur
        self._us_win.append(self._us_proc[i])

        pred_val = np.nan
        if len(self._us_win) == self._W:
            x = torch.from_numpy(
                np.stack(self._us_win)[:, None, :]
            ).unsqueeze(0).to(self._device)
            with torch.no_grad():
                pred_val = self._model(x).item() * self._t_std + self._t_mean

        # 滚动缓冲
        self._pred_roll  = np.roll(self._pred_roll,  -1)
        self._truth_roll = np.roll(self._truth_roll, -1)
        self._us_disp    = np.roll(self._us_disp,    -1, axis=0)
        self._pred_roll[-1]   = pred_val
        self._truth_roll[-1]  = self._truth[i]
        self._us_disp[-1]     = self._us_proc[i]

        # 全程积累
        if not np.isnan(pred_val):
            self._pred_all.append(pred_val)
            self._truth_all.append(float(self._truth[i]))

        self._draw()
        self._cur += 1

    def _draw(self):
        x_roll = np.arange(HISTORY)

        # B-mode
        img_data = (self._us_disp * 255).astype(np.uint8)
        self._img.setImage(img_data, autoLevels=False, levels=(0, 255))
        self._img.setRect(pg.QtCore.QRectF(0, 0, HISTORY, US_N))

        # 力矩叠加在 B-mode（映射到深度范围）
        t_min = float(self._truth_roll.min())
        t_max = float(self._truth_roll.max())
        span  = t_max - t_min if t_max != t_min else 1.0
        def _map(a): return 30.0 + (a - t_min) / span * (US_N - 60)

        self._c_truth_us.setData(x_roll, _map(self._truth_roll))
        valid = ~np.isnan(self._pred_roll)
        if valid.any():
            self._c_pred_us.setData(x_roll[valid], _map(self._pred_roll[valid]))

        # 右上滚动折线
        self._c_truth_roll.setData(x_roll, self._truth_roll)
        if valid.any():
            self._c_pred_roll.setData(x_roll[valid], self._pred_roll[valid])

        # 右下全程积累
        if len(self._pred_all) > 1:
            xa = np.arange(len(self._truth_all))
            self._c_truth_full.setData(xa, np.array(self._truth_all))
            self._c_pred_full.setData(xa,  np.array(self._pred_all))

        # 实时指标
        if len(self._pred_all) > 10:
            pa = np.array(self._pred_all)
            ta = np.array(self._truth_all)
            r2   = _r2(ta, pa)
            rmse = float(np.sqrt(np.mean((pa - ta) ** 2)))
            self._lbl_metric.setText(
                f"R² = {r2:.4f}    RMSE = {rmse:.4f} Nm    帧 {self._cur}/{self._N}")

    def _on_done(self):
        if not self._pred_all:
            return
        pa   = np.array(self._pred_all)
        ta   = np.array(self._truth_all)
        r2   = _r2(ta, pa)
        rmse = float(np.sqrt(np.mean((pa - ta) ** 2)))
        mae  = float(np.mean(np.abs(pa - ta)))

        self._lbl_metric.setText(
            f"✓ 完成    R² = {r2:.4f}    RMSE = {rmse:.4f} Nm    MAE = {mae:.4f} Nm")
        self._lbl_status.setText(
            f"推理完成  总有效帧: {len(pa)}  |  "
            f"R² = {r2:.4f}    RMSE = {rmse:.4f} Nm    MAE = {mae:.4f} Nm")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="UltraTCN 推理对比")
    p.add_argument("--csv",  default="",                  help="标注 CSV 路径")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT),   help="权重文件路径")
    p.add_argument("--fps",  type=float, default=30.0,    help="播放帧率")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app    = QApplication(sys.argv)
    win    = InferCompare(device,
                          csv_path  = args.csv,
                          ckpt_path = args.ckpt,
                          fps       = args.fps)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
