#!/usr/bin/env python3
"""
ultrasound_viewer.py — 超声灰度图单独查看器
从 CSV 文件读取超声数据，全屏显示指定通道的灰度图。

用法：
    python ultrasound_viewer.py
    python ultrasound_viewer.py --file ./data/ultrasound_xxx.csv --channel 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

pg.setConfigOptions(antialias=False, background="w", foreground="k")


def _gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


class UltrasoundViewer(QWidget):
    def __init__(self, csv_file: str = "", channel: int = 1):
        super().__init__()
        self.setWindowTitle("超声灰度图查看器")
        self.resize(1200, 800)

        self._frames: np.ndarray = np.empty((0, 1000), dtype=np.float32)
        self._n_frames = 0
        self._cur_frame = 0
        self._playing = False
        self._history = 300
        self._buf: np.ndarray = np.zeros((self._history, 1000), dtype=np.float32)
        self._channels: list = []

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._build_ui()

        if csv_file:
            self._load_file(csv_file, channel)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── 顶部工具栏 ────────────────────────────────────────────────────────
        top = QHBoxLayout()
        btn_open = QPushButton("打开 CSV"); btn_open.setFixedWidth(90)
        btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("未选择文件"); self._lbl_file.setStyleSheet("color:gray")

        top.addWidget(btn_open)
        top.addWidget(self._lbl_file, stretch=1)

        top.addWidget(QLabel("通道:"))
        self._cb_ch = QComboBox(); self._cb_ch.setFixedWidth(70)
        self._cb_ch.currentIndexChanged.connect(self._on_channel_changed)
        top.addWidget(self._cb_ch)

        top.addWidget(QLabel("显示帧数:"))
        self._spin_hist = QSpinBox()
        self._spin_hist.setRange(10, 5000); self._spin_hist.setValue(300); self._spin_hist.setFixedWidth(70)
        self._spin_hist.valueChanged.connect(self._on_hist_changed)
        top.addWidget(self._spin_hist)

        top.addWidget(QLabel("采样点:"))
        self._spin_s0 = QSpinBox(); self._spin_s0.setRange(0, 999);  self._spin_s0.setValue(0);    self._spin_s0.setFixedWidth(60)
        self._spin_s1 = QSpinBox(); self._spin_s1.setRange(1, 1000); self._spin_s1.setValue(1000); self._spin_s1.setFixedWidth(60)
        btn_s = QPushButton("应用"); btn_s.setFixedWidth(45); btn_s.clicked.connect(self._redraw)
        top.addWidget(self._spin_s0); top.addWidget(QLabel("~")); top.addWidget(self._spin_s1); top.addWidget(btn_s)

        top.addWidget(QLabel("灰度:"))
        self._spin_vmin = QDoubleSpinBox(); self._spin_vmin.setRange(0, 1e6); self._spin_vmin.setValue(0);   self._spin_vmin.setDecimals(0); self._spin_vmin.setFixedWidth(70)
        self._spin_vmax = QDoubleSpinBox(); self._spin_vmax.setRange(0, 1e6); self._spin_vmax.setValue(255); self._spin_vmax.setDecimals(0); self._spin_vmax.setFixedWidth(70)
        btn_v = QPushButton("应用"); btn_v.setFixedWidth(45); btn_v.clicked.connect(self._redraw)
        btn_auto = QPushButton("自动"); btn_auto.setFixedWidth(45); btn_auto.clicked.connect(self._auto_contrast)
        top.addWidget(self._spin_vmin); top.addWidget(QLabel("~")); top.addWidget(self._spin_vmax)
        top.addWidget(btn_v); top.addWidget(btn_auto)

        root.addLayout(top)

        # ── 图像区域（占满剩余空间）──────────────────────────────────────────
        self._pw = pg.PlotWidget()
        self._pw.setLabel("left", "采样点", **{"font-size": "10px"})
        self._pw.setLabel("bottom", "帧", **{"font-size": "10px"})
        self._pw.getAxis("left").setWidth(42)
        self._pw.getAxis("bottom").setHeight(20)
        self._pw.invertY(True)
        self._img = pg.ImageItem()
        self._img.setLookupTable(_gray_lut())
        self._pw.addItem(self._img)
        root.addWidget(self._pw, stretch=1)

        # ── 播放控制 ──────────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._btn_play  = QPushButton("▶ 播放"); self._btn_play.setFixedWidth(80); self._btn_play.setEnabled(False)
        self._btn_pause = QPushButton("⏸ 暂停"); self._btn_pause.setFixedWidth(80); self._btn_pause.setEnabled(False)
        self._btn_stop  = QPushButton("⏹ 停止"); self._btn_stop.setFixedWidth(80)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(0.1, 500); self._spin_fps.setValue(10)
        self._spin_fps.setDecimals(1); self._spin_fps.setFixedWidth(75); self._spin_fps.setSuffix(" fps")
        self._spin_fps.valueChanged.connect(
            lambda v: self._play_timer.setInterval(max(1, int(1000 / v))) if self._playing else None
        )

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0); self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider)

        self._lbl_frame = QLabel("帧: 0 / 0"); self._lbl_frame.setFixedWidth(110)

        ctrl.addWidget(self._btn_play); ctrl.addWidget(self._btn_pause); ctrl.addWidget(self._btn_stop)
        ctrl.addSpacing(12); ctrl.addWidget(QLabel("速度:")); ctrl.addWidget(self._spin_fps)
        ctrl.addSpacing(12); ctrl.addWidget(self._lbl_frame)
        ctrl.addWidget(self._slider, stretch=1)
        root.addLayout(ctrl)

    # ── 文件加载 ──────────────────────────────────────────────────────────────
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择超声 CSV", str(Path.cwd()), "CSV (*.csv)")
        if path:
            self._load_file(path, None)

    def _load_file(self, path: str, channel):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self._lbl_file.setText(f"加载失败: {e}"); return

        dcols = [c for c in df.columns if c.startswith("d")]
        if not dcols or "channel" not in df.columns:
            self._lbl_file.setText("格式错误：缺少 channel 或 d* 列"); return

        self._df    = df
        self._dcols = dcols
        self._channels = sorted(df["channel"].unique().tolist())

        self._cb_ch.blockSignals(True)
        self._cb_ch.clear()
        for ch in self._channels:
            self._cb_ch.addItem(str(ch))
        target = str(channel) if channel is not None else str(self._channels[0])
        idx = self._cb_ch.findText(target)
        self._cb_ch.setCurrentIndex(max(0, idx))
        self._cb_ch.blockSignals(False)

        self._lbl_file.setText(Path(path).name)
        self._lbl_file.setStyleSheet("color:black")
        self._apply_channel()

    def _apply_channel(self):
        ch = int(self._cb_ch.currentText()) if self._cb_ch.count() else 1
        sub = self._df[self._df["channel"] == ch].reset_index(drop=True)
        self._frames   = sub[self._dcols].values.astype(np.float32)
        self._n_frames = len(self._frames)
        self._cur_frame = 0
        self._reset_buf()
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._btn_play.setEnabled(self._n_frames > 0)
        # 自动用全局 min/max 初始化灰度范围，保证最大对比度
        if self._n_frames > 0:
            gmin = float(self._frames.min())
            gmax = float(self._frames.max())
            if gmax <= gmin:
                gmax = gmin + 1
            self._spin_vmin.setValue(gmin)
            self._spin_vmax.setValue(gmax)
        self._render(0)

    def _on_channel_changed(self, _):
        if hasattr(self, "_df"):
            self._on_stop()
            self._apply_channel()

    # ── 缓冲 & 渲染 ───────────────────────────────────────────────────────────
    def _reset_buf(self):
        self._buf = np.zeros((self._history, 1000), dtype=np.float32)

    def _render(self, frame_idx: int):
        if self._n_frames == 0 or frame_idx >= self._n_frames:
            return
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = self._frames[frame_idx]

        vmin = float(self._spin_vmin.value()); vmax = float(self._spin_vmax.value())
        if vmax <= vmin: vmax = vmin + 1
        s0 = max(0, int(self._spin_s0.value())); s1 = min(1000, int(self._spin_s1.value()))
        if s1 <= s0: s1 = s0 + 1

        sl  = self._buf[:, s0:s1]
        img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
        self._img.setImage(img, autoLevels=False, levels=(0, 255))
        self._img.setRect(pg.QtCore.QRectF(0, s0, self._history, s1 - s0))

        self._lbl_frame.setText(f"帧: {frame_idx} / {self._n_frames - 1}")
        self._slider.setValue(frame_idx)
        self._cur_frame = frame_idx

    def _redraw(self):
        if self._n_frames > 0:
            # 重建缓冲以反映新的参数
            start = max(0, self._cur_frame - self._history + 1)
            self._reset_buf()
            for i in range(start, self._cur_frame + 1):
                self._buf = np.roll(self._buf, -1, axis=0)
                self._buf[-1] = self._frames[i]
            vmin = float(self._spin_vmin.value()); vmax = float(self._spin_vmax.value())
            if vmax <= vmin: vmax = vmin + 1
            s0 = max(0, int(self._spin_s0.value())); s1 = min(1000, int(self._spin_s1.value()))
            if s1 <= s0: s1 = s0 + 1
            sl  = self._buf[:, s0:s1]
            img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
            self._img.setImage(img, autoLevels=False, levels=(0, 255))
            self._img.setRect(pg.QtCore.QRectF(0, s0, self._history, s1 - s0))

    def _auto_contrast(self):
        """用当前缓冲区的实际 min/max 自动设置灰度范围。"""
        if self._n_frames == 0:
            return
        s0 = max(0, int(self._spin_s0.value())); s1 = min(1000, int(self._spin_s1.value()))
        if s1 <= s0: s1 = s0 + 1
        sl = self._buf[:, s0:s1]
        vmin = float(sl.min())
        vmax = float(sl.max())
        if vmax <= vmin:
            vmax = vmin + 1
        self._spin_vmin.setValue(vmin)
        self._spin_vmax.setValue(vmax)
        self._redraw()

    # ── 播放控制 ──────────────────────────────────────────────────────────────
    def _on_play(self):
        if not self._n_frames: return
        if self._cur_frame >= self._n_frames - 1:
            self._cur_frame = 0; self._reset_buf()
        self._playing = True
        self._btn_play.setEnabled(False); self._btn_pause.setEnabled(True)
        self._play_timer.start(max(1, int(1000 / self._spin_fps.value())))

    def _on_pause(self):
        self._playing = False; self._play_timer.stop()
        self._btn_play.setEnabled(True); self._btn_pause.setEnabled(False)

    def _on_stop(self):
        self._playing = False; self._play_timer.stop()
        self._btn_play.setEnabled(bool(self._n_frames)); self._btn_pause.setEnabled(False)
        self._cur_frame = 0; self._reset_buf()
        if self._n_frames: self._render(0)

    def _next_frame(self):
        if self._cur_frame >= self._n_frames - 1:
            self._on_stop(); return
        self._render(self._cur_frame + 1)

    def _on_slider(self, val: int):
        self._on_pause()
        self._reset_buf()
        for i in range(max(0, val - self._history + 1), val + 1):
            self._render(i)

    def _on_hist_changed(self, val: int):
        self._history = val
        self._reset_buf()
        if self._n_frames:
            self._render(self._cur_frame)


def main():
    parser = argparse.ArgumentParser(description="超声灰度图查看器")
    parser.add_argument("--file",    default="", help="CSV 文件路径")
    parser.add_argument("--channel", type=int, default=1, help="默认显示通道（默认 1）")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = UltrasoundViewer(csv_file=args.file, channel=args.channel)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
