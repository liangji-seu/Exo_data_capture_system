#!/usr/bin/env python3
"""
ultrasound_viewer.py — 超声灰度图单独查看器
从 CSV 文件读取超声数据，同时显示全部通道的灰度图（2x2 布局）。

用法：
    python ultrasound_viewer.py
    python ultrasound_viewer.py --file ./data/ultrasound_xxx.csv
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
        QSpinBox, QDoubleSpinBox,
    )
except ImportError:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog,
        QSpinBox, QDoubleSpinBox,
    )

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="k", foreground="w")

_CHANNELS = [1, 2, 3, 4]


def _gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


class UltrasoundViewer(QWidget):
    def __init__(self, csv_file: str = ""):
        super().__init__()
        self.setWindowTitle("超声灰度图查看器 — 4通道")
        self.resize(1200, 800)

        # per-channel state
        self._frames: dict[int, np.ndarray] = {}   # ch -> (N, 1000)
        self._bufs:   dict[int, np.ndarray] = {}   # ch -> (history, 1000)
        self._n_frames = 0
        self._cur_frame = 0
        self._playing = False
        self._history = 300

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._img_items: dict[int, pg.ImageItem] = {}

        self._build_ui()

        if csv_file:
            self._load_file(csv_file)

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
        top.addWidget(self._spin_vmin); top.addWidget(QLabel("~")); top.addWidget(self._spin_vmax); top.addWidget(btn_v)

        top.addWidget(QLabel("黑点%:"))
        self._spin_plow = QSpinBox(); self._spin_plow.setRange(0, 99); self._spin_plow.setValue(85); self._spin_plow.setFixedWidth(48)
        top.addWidget(self._spin_plow)

        top.addWidget(QLabel("统计范围:"))
        self._spin_ps0 = QSpinBox(); self._spin_ps0.setRange(0, 999);  self._spin_ps0.setValue(0);    self._spin_ps0.setFixedWidth(55)
        self._spin_ps1 = QSpinBox(); self._spin_ps1.setRange(1, 1000); self._spin_ps1.setValue(1000); self._spin_ps1.setFixedWidth(55)
        top.addWidget(self._spin_ps0); top.addWidget(QLabel("~")); top.addWidget(self._spin_ps1)

        btn_auto = QPushButton("自动"); btn_auto.setFixedWidth(45); btn_auto.clicked.connect(self._auto_contrast)
        top.addWidget(btn_auto)

        root.addLayout(top)

        # ── 4通道图像区域（2×2）──────────────────────────────────────────────
        lut = _gray_lut()
        grid_top = QHBoxLayout(); grid_top.setSpacing(4)
        grid_bot = QHBoxLayout(); grid_bot.setSpacing(4)
        for ch in _CHANNELS:
            pw = pg.PlotWidget()
            pw.setLabel("left",   f"ch{ch} 采样点", **{"font-size": "9px"})
            pw.setLabel("bottom", "帧",              **{"font-size": "9px"})
            pw.getAxis("left").setWidth(42)
            pw.getAxis("bottom").setHeight(18)
            pw.invertY(True)
            img = pg.ImageItem()
            img.setLookupTable(lut)
            pw.addItem(img)
            self._img_items[ch] = img
            (grid_top if ch <= 2 else grid_bot).addWidget(pw)

        img_container = QWidget()
        img_ly = QVBoxLayout(img_container)
        img_ly.setContentsMargins(0, 0, 0, 0); img_ly.setSpacing(4)
        img_ly.addLayout(grid_top)
        img_ly.addLayout(grid_bot)
        root.addWidget(img_container, stretch=1)

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
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self._lbl_file.setText(f"加载失败: {e}"); return

        dcols = [c for c in df.columns if c.startswith("d")]
        if not dcols or "channel" not in df.columns:
            self._lbl_file.setText("格式错误：缺少 channel 或 d* 列"); return

        channels = sorted(df["channel"].unique().tolist())
        self._frames = {}
        for ch in channels:
            sub = df[df["channel"] == ch].reset_index(drop=True)
            self._frames[int(ch)] = sub[dcols].values.astype(np.float32)

        # 以第一个通道的帧数为基准
        first = min(self._frames)
        self._n_frames  = len(self._frames[first])
        self._cur_frame = 0
        self._reset_bufs()
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._btn_play.setEnabled(self._n_frames > 0)

        self._lbl_file.setText(Path(path).name)
        self._lbl_file.setStyleSheet("color:white")

        # 用所有通道数据的指定统计范围合并计算百分位，统一灰度范围
        ps0 = max(0, int(self._spin_ps0.value())); ps1 = min(1000, int(self._spin_ps1.value()))
        if ps1 <= ps0: ps1 = ps0 + 1
        all_data = np.concatenate([v[:, ps0:ps1] for v in self._frames.values()])
        self._apply_percentile_contrast(all_data)
        self._render(0)

    # ── 缓冲 & 渲染 ───────────────────────────────────────────────────────────
    def _reset_bufs(self):
        self._bufs = {ch: np.zeros((self._history, 1000), dtype=np.float32)
                      for ch in self._frames}

    def _render(self, frame_idx: int):
        if not self._frames or frame_idx >= self._n_frames:
            return

        vmin = float(self._spin_vmin.value()); vmax = float(self._spin_vmax.value())
        if vmax <= vmin: vmax = vmin + 1
        s0 = max(0, int(self._spin_s0.value())); s1 = min(1000, int(self._spin_s1.value()))
        if s1 <= s0: s1 = s0 + 1

        for ch, frames in self._frames.items():
            buf = self._bufs[ch]
            if frame_idx < len(frames):
                buf = np.roll(buf, -1, axis=0)
                buf[-1] = frames[frame_idx]
                self._bufs[ch] = buf
            sl  = buf[:, s0:s1]
            img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
            item = self._img_items.get(ch)
            if item:
                item.setImage(img, autoLevels=False, levels=(0, 255))
                item.setRect(pg.QtCore.QRectF(0, s0, self._history, s1 - s0))

        self._lbl_frame.setText(f"帧: {frame_idx} / {self._n_frames - 1}")
        self._slider.setValue(frame_idx)
        self._cur_frame = frame_idx

    def _redraw(self):
        if not self._frames:
            return
        start = max(0, self._cur_frame - self._history + 1)
        self._reset_bufs()
        for i in range(start, self._cur_frame + 1):
            for ch, frames in self._frames.items():
                if i < len(frames):
                    self._bufs[ch] = np.roll(self._bufs[ch], -1, axis=0)
                    self._bufs[ch][-1] = frames[i]

        vmin = float(self._spin_vmin.value()); vmax = float(self._spin_vmax.value())
        if vmax <= vmin: vmax = vmin + 1
        s0 = max(0, int(self._spin_s0.value())); s1 = min(1000, int(self._spin_s1.value()))
        if s1 <= s0: s1 = s0 + 1

        for ch, buf in self._bufs.items():
            sl  = buf[:, s0:s1]
            img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
            item = self._img_items.get(ch)
            if item:
                item.setImage(img, autoLevels=False, levels=(0, 255))
                item.setRect(pg.QtCore.QRectF(0, s0, self._history, s1 - s0))

    def _auto_contrast(self):
        if not self._frames:
            return
        ps0 = max(0, int(self._spin_ps0.value())); ps1 = min(1000, int(self._spin_ps1.value()))
        if ps1 <= ps0: ps1 = ps0 + 1
        # 用所有通道当前缓冲区的指定统计范围合并计算
        all_buf = np.concatenate([buf[:, ps0:ps1] for buf in self._bufs.values()])
        self._apply_percentile_contrast(all_buf)
        self._redraw()

    def _apply_percentile_contrast(self, data: np.ndarray):
        plow = int(self._spin_plow.value())
        vmin = float(np.percentile(data, plow))
        vmax = float(np.percentile(data, 99.9))
        if vmax <= vmin:
            vmax = vmin + 1
        self._spin_vmin.setValue(vmin)
        self._spin_vmax.setValue(vmax)

    # ── 播放控制 ──────────────────────────────────────────────────────────────
    def _on_play(self):
        if not self._n_frames: return
        if self._cur_frame >= self._n_frames - 1:
            self._cur_frame = 0; self._reset_bufs()
        self._playing = True
        self._btn_play.setEnabled(False); self._btn_pause.setEnabled(True)
        self._play_timer.start(max(1, int(1000 / self._spin_fps.value())))

    def _on_pause(self):
        self._playing = False; self._play_timer.stop()
        self._btn_play.setEnabled(True); self._btn_pause.setEnabled(False)

    def _on_stop(self):
        self._playing = False; self._play_timer.stop()
        self._btn_play.setEnabled(bool(self._n_frames)); self._btn_pause.setEnabled(False)
        self._cur_frame = 0; self._reset_bufs()
        if self._n_frames: self._render(0)

    def _next_frame(self):
        if self._cur_frame >= self._n_frames - 1:
            self._on_stop(); return
        self._render(self._cur_frame + 1)

    def _on_slider(self, val: int):
        self._on_pause()
        self._reset_bufs()
        for i in range(max(0, val - self._history + 1), val + 1):
            self._render(i)

    def _on_hist_changed(self, val: int):
        self._history = val
        self._reset_bufs()
        if self._n_frames:
            self._render(self._cur_frame)


def main():
    parser = argparse.ArgumentParser(description="超声灰度图查看器")
    parser.add_argument("--file", default="", help="CSV 文件路径")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = UltrasoundViewer(csv_file=args.file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
