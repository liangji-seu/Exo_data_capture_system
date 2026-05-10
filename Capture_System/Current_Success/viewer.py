#!/usr/bin/env python3
"""
viewer.py — Current_Success 数据集可视化
数据格式: Feat_1..Feat_100 (超声幅值) + Label_Sin, Label_Cos (步态相位)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog,
        QSpinBox, QDoubleSpinBox, QComboBox,
    )
except ImportError:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog,
        QSpinBox, QDoubleSpinBox, QComboBox,
    )

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="k", foreground="w")

_DEFAULT_DIR = str(Path(__file__).parent / "数据集文件" / "Processed_Final_Training_Set_clean_v2_fix_shift_v2")


def _gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


def reconstruct_phase_deg(sin_vals: np.ndarray, cos_vals: np.ndarray) -> np.ndarray:
    """sin/cos → 0~360° 步态相位角"""
    return (np.degrees(np.arctan2(sin_vals, cos_vals)) + 360) % 360


def normalize_ultrasound(feat: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """对 (N, 100) 超声幅值做归一化 + 轻度平滑，返回 float32 [0,1]"""
    f = feat.astype(np.float32)
    # 逐帧自适应拉伸
    vmin = np.percentile(f, 5,  axis=1, keepdims=True)
    vmax = np.percentile(f, 99, axis=1, keepdims=True)
    f = np.clip((f - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)
    f = median_filter(f, size=(3, 3))
    if sigma > 0:
        f = gaussian_filter(f, sigma=[sigma, 0.5])
    return f.astype(np.float32)


class DatasetViewer(QWidget):
    def __init__(self, csv_file=""):
        super().__init__()
        self.setWindowTitle("Current_Success 数据集可视化")
        self.resize(1100, 700)

        self._feat:      np.ndarray = np.empty((0, 100), dtype=np.float32)
        self._proc:      np.ndarray = np.empty((0, 100), dtype=np.float32)
        self._phase_deg: np.ndarray = np.array([])
        self._n_frames   = 0
        self._cur_frame  = 0
        self._playing    = False
        self._history    = 300

        self._phase_buf: np.ndarray = np.zeros(300)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._build_ui()

        if csv_file:
            self._load_file(csv_file)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # 顶栏：文件操作
        top = QHBoxLayout()
        btn_open = QPushButton("打开 CSV"); btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("待载入...")
        top.addWidget(btn_open)
        top.addWidget(self._lbl_file, stretch=1)

        top.addWidget(QLabel("时间轴(帧):"))
        self._spin_hist = QSpinBox()
        self._spin_hist.setRange(50, 2000); self._spin_hist.setValue(300)
        self._spin_hist.valueChanged.connect(self._on_hist_changed)
        top.addWidget(self._spin_hist)

        top.addWidget(QLabel("平滑σ:"))
        self._spin_sigma = QDoubleSpinBox()
        self._spin_sigma.setRange(0, 5); self._spin_sigma.setValue(0.8); self._spin_sigma.setDecimals(1)
        top.addWidget(self._spin_sigma)

        btn_reproc = QPushButton("刷新"); btn_reproc.clicked.connect(self._reprocess)
        top.addWidget(btn_reproc)
        root.addLayout(top)

        # 主图区：上=超声瀑布图，下=步态相位曲线
        lut = _gray_lut()

        self._pw = pg.PlotWidget(title="超声幅值 (Feat_1~100)")
        self._pw.invertY(False)
        self._pw.getAxis("bottom").setLabel("")
        self._pw.getAxis("left").setLabel("特征索引")

        self._img_item = pg.ImageItem()
        self._img_item.setLookupTable(lut)
        self._img_item.setZValue(0)
        self._pw.addItem(self._img_item)

        # 步态周期竖线（叠在超声图上）
        self._cycle_line = pg.PlotDataItem(pen=pg.mkPen("#FF0000", width=1.5))
        self._cycle_line.setZValue(11)
        self._pw.addItem(self._cycle_line)

        # 步态相位独立子图
        self._pw_phase = pg.PlotWidget(title="步态相位 (°)")
        self._pw_phase.setMaximumHeight(160)
        self._pw_phase.getAxis("bottom").setLabel("帧 (滚动)")
        self._pw_phase.getAxis("left").setLabel("相位 (°)")
        self._pw_phase.setYRange(0, 360)
        self._pw_phase.setXLink(self._pw)

        self._phase_curve = self._pw_phase.plot(pen=pg.mkPen("#FF5722", width=2))

        # 步态周期竖线（相位图里也画）
        self._cycle_line_phase = pg.PlotDataItem(pen=pg.mkPen("#FF0000", width=1.5))
        self._pw_phase.addItem(self._cycle_line_phase)

        root.addWidget(self._pw, stretch=1)
        root.addWidget(self._pw_phase)

        # 信息栏
        info = QHBoxLayout()
        self._lbl_phase = QLabel("相位: --°")
        self._lbl_phase.setStyleSheet("color:#FF5722; font-size:13px")
        self._lbl_sin   = QLabel("Sin: --  Cos: --")
        self._lbl_sin.setStyleSheet("color:#888; font-size:12px")
        info.addWidget(self._lbl_phase)
        info.addSpacing(20)
        info.addWidget(self._lbl_sin)
        info.addStretch()
        root.addLayout(info)

        # 播放控制
        bot = QHBoxLayout()
        self._btn_play  = QPushButton("▶ 播放");  self._btn_play.clicked.connect(self._on_play)
        self._btn_pause = QPushButton("⏸ 暂停"); self._btn_pause.clicked.connect(self._on_pause)
        self._slider    = QSlider(Qt.Horizontal); self._slider.sliderMoved.connect(self._on_slider)
        self._lbl_frame = QLabel("帧: 0/0")
        self._spin_fps  = QDoubleSpinBox()
        self._spin_fps.setRange(1, 200); self._spin_fps.setValue(30); self._spin_fps.setSuffix(" fps")
        self._spin_fps.setFixedWidth(80)
        self._spin_fps.valueChanged.connect(
            lambda v: self._play_timer.setInterval(max(1, int(1000 / v))) if self._playing else None)
        bot.addWidget(self._btn_play); bot.addWidget(self._btn_pause)
        bot.addSpacing(8); bot.addWidget(QLabel("速度:")); bot.addWidget(self._spin_fps)
        bot.addSpacing(8); bot.addWidget(self._lbl_frame)
        bot.addWidget(self._slider, stretch=1)
        root.addLayout(bot)


    # ── 文件加载 ──────────────────────────────────────────────────────────
    def _on_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择数据集 CSV", _DEFAULT_DIR, "CSV (*.csv)")
        if p:
            self._load_file(p)

    def _load_file(self, path: str):
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns if c.startswith("Feat_")]
        if not feat_cols:
            self._lbl_file.setText("格式错误：找不到 Feat_ 列"); return

        self._feat      = df[feat_cols].to_numpy(dtype=np.float32)
        sin_vals        = df["Label_Sin"].to_numpy(dtype=np.float64) if "Label_Sin" in df.columns else np.zeros(len(df))
        cos_vals        = df["Label_Cos"].to_numpy(dtype=np.float64) if "Label_Cos" in df.columns else np.zeros(len(df))
        self._phase_deg = reconstruct_phase_deg(sin_vals, cos_vals)

        self._lbl_file.setText(f"{Path(path).name}  ({len(df)} 帧, {len(feat_cols)} 特征)")
        self._reprocess()

    def _reprocess(self):
        if self._feat.size == 0: return
        self._proc     = normalize_ultrasound(self._feat, self._spin_sigma.value())
        self._n_frames = len(self._proc)
        self._slider.setMaximum(self._n_frames - 1)
        self._reset_bufs()
        self._render(0)

    # ── 缓冲 & 渲染 ───────────────────────────────────────────────────────
    def _reset_bufs(self):
        n_feat = self._proc.shape[1] if self._proc.ndim == 2 else 100
        self._buf       = np.zeros((self._history, n_feat), dtype=np.float32)
        self._phase_buf = np.zeros(self._history)

    def _render(self, idx: int):
        if self._proc.size == 0: return
        x_axis = np.arange(self._history)

        # 推入滚动缓冲
        self._buf       = np.roll(self._buf,       -1, axis=0)
        self._buf[-1]   = self._proc[idx]
        self._phase_buf = np.roll(self._phase_buf, -1)
        self._phase_buf[-1] = float(self._phase_deg[idx]) if idx < len(self._phase_deg) else 0.0

        # 超声图像
        img_data = (self._buf * 255).astype(np.uint8)
        n_feat   = img_data.shape[1]
        self._img_item.setImage(img_data, autoLevels=False, levels=(0, 255))
        self._img_item.setRect(pg.QtCore.QRectF(0, 0, self._history, n_feat))

        # 步态相位曲线（独立子图）
        self._phase_curve.setData(x_axis, self._phase_buf)

        # 步态周期竖线
        cycle_xs, cycle_ys_img, cycle_ys_phase = [], [], []
        for i in range(1, len(self._phase_buf)):
            if self._phase_buf[i - 1] > 300 and self._phase_buf[i] < 60:
                cycle_xs += [i, i, float('nan')]
                cycle_ys_img   += [0, n_feat, float('nan')]
                cycle_ys_phase += [0, 360,    float('nan')]
        self._cycle_line.setData(cycle_xs, cycle_ys_img)
        self._cycle_line_phase.setData(cycle_xs, cycle_ys_phase)

        # 信息栏
        phase_val = self._phase_deg[idx] if idx < len(self._phase_deg) else 0.0
        self._lbl_phase.setText(f"相位: {phase_val:.1f}°")
        if idx < len(self._feat):
            sin_v = np.sin(np.radians(phase_val))
            cos_v = np.cos(np.radians(phase_val))
            self._lbl_sin.setText(f"Sin: {sin_v:.3f}  Cos: {cos_v:.3f}")

        self._lbl_frame.setText(f"帧: {idx}/{self._n_frames}")
        self._slider.setValue(idx)
        self._cur_frame = idx

    # ── 播放控制 ──────────────────────────────────────────────────────────
    def _on_play(self):
        self._playing = True
        self._play_timer.start(max(1, int(1000 / self._spin_fps.value())))

    def _on_pause(self):
        self._playing = False
        self._play_timer.stop()

    def _next_frame(self):
        if self._cur_frame < self._n_frames - 1:
            self._render(self._cur_frame + 1)
        else:
            self._on_pause()

    def _on_slider(self, v):
        self._on_pause()
        self._reset_bufs()
        self._render(v)

    def _on_hist_changed(self, v):
        self._history = v
        self._reset_bufs()
        self._render(self._cur_frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="", help="数据集 CSV 路径")
    args = parser.parse_args()
    app  = QApplication(sys.argv)
    win  = DatasetViewer(csv_file=args.file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
