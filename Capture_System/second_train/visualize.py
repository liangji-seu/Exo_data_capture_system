#!/usr/bin/env python3
"""
visualize.py — 单通道超声 TCN 流式推理可视化

用法：
  python visualize.py
  python visualize.py --csv data/20260422_141047_1.csv --ckpt checkpoints/best.pt
  python visualize.py --fps 60

界面：
  上方  B-mode 瀑布图 + 预测/真值曲线叠加
  下方  力矩折线图（红=预测，蓝=真值），Y 轴为真实 Nm
  右侧  实时 MAE / RMSE 指标
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

pg.setConfigOptions(antialias=False, background="k", foreground="w")

sys.path.insert(0, str(Path(__file__).parent))
from model import UltraTCN

HISTORY = 400   # 滚动显示帧数
US_N    = 300   # ROI 采样点数

DEFAULT_CKPT = Path(__file__).parent / "checkpoints" / "best.pt"
DEFAULT_DATA = Path(__file__).parent / "data"


# ── 预处理 ────────────────────────────────────────────────────────────────────

def _preprocess(raw: np.ndarray,
                tgc_slope=0.025, sigma_depth=1.2,
                sigma_time=0.8,  sharpen=0.2) -> np.ndarray:
    """raw: (N, 300) → (N, 300) float32 [0, 1]"""
    raw_f    = raw.astype(np.float32)
    n, nsamp = raw_f.shape
    rf       = raw_f - raw_f.mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(rf, axis=1))
    log_env  = 20.0 * np.log10(np.clip(envelope, 1e-6, None))
    out      = np.zeros_like(log_env)
    tgc_gain = np.arange(nsamp) * tgc_slope
    for i in range(n):
        f      = log_env[i] + tgc_gain
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


# ── 模型加载 ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
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

class InferViewer(QWidget):
    def __init__(self, device: torch.device,
                 csv_path: str = "", ckpt_path: str = "", fps: float = 30.0):
        super().__init__()
        self.setWindowTitle("UltraTCN 流式推理 — second_train")
        self.resize(1300, 820)

        self._device    = device
        self._model     = None
        self._t_mean    = 0.0
        self._t_std     = 1.0
        self._W         = 50

        self._us_proc:  np.ndarray = np.array([])   # (N, 300) 预处理后
        self._truth:    np.ndarray = np.array([])   # (N,) 真值 Nm
        self._N         = 0
        self._cur       = 0
        self._playing   = False

        # 流式推理缓冲（滑窗）
        self._us_win: deque = deque()

        # 滚动显示缓冲
        self._us_disp   = np.zeros((HISTORY, US_N), dtype=np.float32)
        self._pred_buf  = np.full(HISTORY, np.nan, dtype=np.float32)
        self._truth_buf = np.zeros(HISTORY, dtype=np.float32)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)

        self._build_ui()

        if ckpt_path:
            self._load_ckpt(ckpt_path)
        if csv_path:
            self._load_csv(csv_path)
        self._set_fps(fps)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # 工具栏
        bar = QHBoxLayout()

        btn_ckpt = QPushButton("📦 加载权重")
        btn_ckpt.clicked.connect(lambda: self._on_open_ckpt())
        self._lbl_ckpt = QLabel("未加载权重")
        self._lbl_ckpt.setStyleSheet("color:#f90")

        btn_csv = QPushButton("📂 打开 CSV")
        btn_csv.clicked.connect(lambda: self._on_open_csv())
        self._lbl_csv = QLabel("未加载数据")
        self._lbl_csv.setStyleSheet("color:#888")

        self._btn_play = QPushButton("▶ 播放")
        self._btn_play.clicked.connect(self._on_play_pause)
        self._btn_play.setEnabled(False)

        btn_reset = QPushButton("⏮ 重置")
        btn_reset.clicked.connect(self._reset)

        fps_lbl = QLabel("FPS:")
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1, 200); self._spin_fps.setValue(30); self._spin_fps.setFixedWidth(70)
        self._spin_fps.valueChanged.connect(self._set_fps)

        self._lbl_stat = QLabel("—")
        self._lbl_stat.setStyleSheet("color:#aaa; font-size:11px")

        for w in [btn_ckpt, self._lbl_ckpt, btn_csv, self._lbl_csv,
                  self._btn_play, btn_reset, fps_lbl, self._spin_fps,
                  self._lbl_stat]:
            bar.addWidget(w)
        bar.addStretch()
        root.addLayout(bar)

        lut = _gray_lut()

        # ── 上方：B-mode 瀑布图 + 叠加曲线 ─────────────────────────────────
        self._pw_us = pg.PlotWidget(title="B-mode 瀑布  （红=预测，蓝=真值）")
        self._pw_us.setLabel("bottom", "帧")
        self._pw_us.setLabel("left",   "深度（ROI 0~300）")
        self._pw_us.setMinimumHeight(280)

        self._img_item = pg.ImageItem()
        self._img_item.setLookupTable(lut)
        self._pw_us.addItem(self._img_item)

        self._curve_truth_us = self._pw_us.plot(pen=pg.mkPen("#4FC3F7", width=2))
        self._curve_pred_us  = self._pw_us.plot(pen=pg.mkPen("#EF5350", width=2))
        self._curve_truth_us.setZValue(10)
        self._curve_pred_us.setZValue(11)

        for text, color, anchor in [("Ground Truth", "#4FC3F7", (0, 1)),
                                     ("Prediction",   "#EF5350", (0, 1))]:
            t = pg.TextItem(text, color=color, anchor=anchor)
            self._pw_us.addItem(t)
            t.setPos(5, 20 if "Ground" in text else 40)
            t.setZValue(20)

        root.addWidget(self._pw_us, stretch=2)

        # ── 下方：力矩折线图（真实 Nm 量纲）────────────────────────────────
        self._pw_tq = pg.PlotWidget(title="髋关节力矩  （红=预测，蓝=真值）")
        self._pw_tq.setLabel("bottom", "帧")
        self._pw_tq.setLabel("left",   "Torque (Nm)")
        self._pw_tq.showGrid(x=True, y=True, alpha=0.25)
        self._pw_tq.addLegend(offset=(10, 10))
        self._pw_tq.setMinimumHeight(220)

        self._curve_truth_tq = self._pw_tq.plot(
            pen=pg.mkPen("#4FC3F7", width=2), name="Ground Truth")
        self._curve_pred_tq  = self._pw_tq.plot(
            pen=pg.mkPen("#EF5350", width=2), name="Prediction")

        root.addWidget(self._pw_tq, stretch=1)

        # 链接 x 轴
        self._pw_tq.setXLink(self._pw_us)

    # ── 文件操作 ──────────────────────────────────────────────────────────────

    def _on_open_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择权重文件", str(DEFAULT_CKPT.parent), "PyTorch (*.pt *.pth)")
        if p:
            self._load_ckpt(p)

    def _load_ckpt(self, path: str):
        try:
            self._model, self._t_mean, self._t_std, self._W = \
                load_model(Path(path), self._device)
            self._us_win = deque(maxlen=self._W)
            self._lbl_ckpt.setText(f"{Path(path).name}  (W={self._W})")
            self._lbl_ckpt.setStyleSheet("color:#4FC3F7")
        except Exception as e:
            self._lbl_ckpt.setText(f"加载失败: {e}")
            self._lbl_ckpt.setStyleSheet("color:#f44")

    def _on_open_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "选择标注 CSV", str(DEFAULT_DATA), "CSV (*.csv)")
        if p:
            self._load_csv(p)

    def _load_csv(self, path: str):
        try:
            df     = pd.read_csv(path)
            dcols  = [f"d{i}" for i in range(US_N)]
            raw    = df[dcols].values.astype(np.float32)
            self._lbl_csv.setText("预处理中…")
            QApplication.processEvents()
            self._us_proc = _preprocess(raw)
            self._truth   = df["torque"].values.astype(np.float32)
            self._N       = len(self._truth)
            self._reset()
            self._lbl_csv.setText(f"{Path(path).name}  ({self._N} 帧)")
            self._lbl_csv.setStyleSheet("color:#aaa")
            if self._model:
                self._btn_play.setEnabled(True)
        except Exception as e:
            self._lbl_csv.setText(f"加载失败: {e}")
            self._lbl_csv.setStyleSheet("color:#f44")

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
        self._cur      = 0
        self._us_win   = deque(maxlen=self._W)
        self._us_disp  = np.zeros((HISTORY, US_N), dtype=np.float32)
        self._pred_buf = np.full(HISTORY, np.nan, dtype=np.float32)
        self._truth_buf= np.zeros(HISTORY, dtype=np.float32)
        self._img_item.clear()
        self._curve_truth_us.setData([], [])
        self._curve_pred_us.setData([], [])
        self._curve_truth_tq.setData([], [])
        self._curve_pred_tq.setData([], [])
        if self._model and self._N:
            self._btn_play.setEnabled(True)

    def _set_fps(self, v: float):
        interval = max(1, int(1000 / v))
        self._timer.setInterval(interval)

    # ── 每帧推理 + 绘图 ───────────────────────────────────────────────────────

    def _step(self):
        if self._model is None or self._N == 0:
            return
        if self._cur >= self._N:
            self._timer.stop()
            self._playing = False
            self._btn_play.setText("▶ 播放")
            return

        i = self._cur

        # 推进滑窗
        self._us_win.append(self._us_proc[i])   # (300,)

        pred_val = np.nan
        if len(self._us_win) == self._W:
            us_tensor = torch.from_numpy(
                np.stack(self._us_win)[:, None, :]   # (W, 1, 300)
            ).unsqueeze(0).to(self._device)          # (1, W, 1, 300)
            with torch.no_grad():
                pred_norm = self._model(us_tensor).item()
            pred_val = pred_norm * self._t_std + self._t_mean

        # 滚动缓冲
        self._pred_buf  = np.roll(self._pred_buf,  -1); self._pred_buf[-1]  = pred_val
        self._truth_buf = np.roll(self._truth_buf, -1); self._truth_buf[-1] = self._truth[i]
        self._us_disp   = np.roll(self._us_disp, -1, axis=0)
        self._us_disp[-1] = self._us_proc[i]

        self._draw()
        self._cur += 1

    def _draw(self):
        x = np.arange(HISTORY)

        # B-mode
        img_data = (self._us_disp * 255).astype(np.uint8)
        self._img_item.setImage(img_data, autoLevels=False, levels=(0, 255))
        self._img_item.setRect(pg.QtCore.QRectF(0, 0, HISTORY, US_N))

        # 把力矩映射到深度范围叠加在 B-mode 上
        t_min = float(self._truth_buf.min())
        t_max = float(self._truth_buf.max())
        span  = t_max - t_min if t_max != t_min else 1.0

        def _map(arr):
            return 50.0 + (arr - t_min) / span * (US_N - 100)

        self._curve_truth_us.setData(x, _map(self._truth_buf))

        valid = ~np.isnan(self._pred_buf)
        if valid.any():
            self._curve_pred_us.setData(x[valid], _map(self._pred_buf[valid]))

        # 力矩折线图（真实量纲）
        self._curve_truth_tq.setData(x, self._truth_buf)
        if valid.any():
            self._curve_pred_tq.setData(x[valid], self._pred_buf[valid])

        # 指标
        if valid.any():
            err  = self._pred_buf[valid] - self._truth_buf[valid]
            mae  = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            self._lbl_stat.setText(
                f"帧 {self._cur}/{self._N}    MAE = {mae:.4f} Nm    RMSE = {rmse:.4f} Nm")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="UltraTCN 流式推理可视化")
    p.add_argument("--csv",  default="", help="标注 CSV 路径（可启动后在 GUI 中打开）")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="权重文件路径")
    p.add_argument("--fps",  type=float, default=30.0,  help="播放帧率")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app    = QApplication(sys.argv)
    win    = InferViewer(device,
                         csv_path  = args.csv,
                         ckpt_path = args.ckpt if Path(args.ckpt).exists() else "",
                         fps       = args.fps)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
