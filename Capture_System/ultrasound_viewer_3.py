#!/usr/bin/env python3
"""
ultrasound_viewer.py — 超声灰度图查看器（4通道 M-mode）
预处理管线（参考 Jin et al., 2024, Nature Communications）：
  1. 带通滤波  4–6 MHz Butterworth
  2. 希尔伯特变换包络提取
  3. 对数压缩 20·log10 + 线性 TGC
  4. 2D 高斯平滑（深度 × 时间）

用法：
    python ultrasound_viewer.py
    python ultrasound_viewer.py --file ./data/ultrasound_xxx.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

try:
    from PySide6.QtCore import QTimer, Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog,
        QSpinBox, QDoubleSpinBox,
    )
except ImportError:
    from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog,
        QSpinBox, QDoubleSpinBox,
    )

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="k", foreground="w")

_CHANNELS    = [1, 2, 3, 4]
_FS          = 100.0        # 帧率 Hz
_SOUND_SPEED = 1540.0       # 声速 m/s
_N_SAMPLES   = 1000


# ─────────────────────────────────────────────────────────────────────────────
#  预处理管线
# ─────────────────────────────────────────────────────────────────────────────

class UltrasoundProcessor:
    """
    对 shape=(N_frames, 1000) 的包络强度数据执行预处理，
    返回 shape=(N_frames, 1000) 的归一化图像，值域 [0, 1]。

    注意：输入数据已是包络强度值（非原始 RF），
    因此跳过带通滤波和希尔伯特变换，直接做：
      1. 对数压缩（20·log10）
      2. TGC 线性深度增益补偿（单位 dB/采样点）
      3. 2D 高斯平滑
    """

    def __init__(self, tgc_slope: float = 0.0,
                 sigma_depth: float = 1.5, sigma_time: float = 1.0,
                 dynamic_range: float = 15.0):
        self.tgc_slope     = tgc_slope      # dB/采样点 深度增益斜率
        self.sigma_depth   = sigma_depth    # 深度方向高斯 sigma（采样点单位）
        self.sigma_time    = sigma_time     # 时间方向高斯 sigma（帧单位）
        self.dynamic_range = dynamic_range  # 动态范围 dB（数据实际范围约 12 dB）

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """
        raw: (N_frames, 1000) float32 包络强度数据
        返回: (N_frames, 1000) float32，归一化到 [0, 1]
        """
        # 1. 对数压缩：压缩动态范围，使弱回波可见
        envelope = np.clip(raw.astype(np.float32), 1e-6, None)
        log_env  = 20.0 * np.log10(envelope)

        # 2. TGC：补偿超声随深度的衰减，深层信号线性增强（dB/采样点）
        if self.tgc_slope != 0.0:
            tgc_gain = np.arange(_N_SAMPLES, dtype=np.float32) * self.tgc_slope
            log_env += tgc_gain[np.newaxis, :]

        # 3. 2D 高斯平滑：深度方向去斑点噪声，时间方向保证边界运动连贯
        if self.sigma_depth > 0 or self.sigma_time > 0:
            smoothed = gaussian_filter(log_env,
                                       sigma=[self.sigma_time, self.sigma_depth])
        else:
            smoothed = log_env

        # 4. 动态范围截断 + 归一化到 [0, 1]
        vmax = smoothed.max()
        vmin = vmax - self.dynamic_range
        out  = np.clip((smoothed - vmin) / self.dynamic_range, 0.0, 1.0)
        return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  后台处理线程（避免 UI 卡顿）
# ─────────────────────────────────────────────────────────────────────────────

class ProcessThread(QThread):
    finished = Signal(dict)   # ch -> processed ndarray

    def __init__(self, raw_frames: dict, processor: UltrasoundProcessor):
        super().__init__()
        self._raw  = raw_frames
        self._proc = processor

    def run(self):
        result = {}
        for ch, frames in self._raw.items():
            result[ch] = self._proc.preprocess(frames)
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
#  灰度 LUT
# ─────────────────────────────────────────────────────────────────────────────

def _gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────

class UltrasoundViewer(QWidget):
    def __init__(self, csv_file: str = ""):
        super().__init__()
        self.setWindowTitle("超声灰度图查看器 — 4通道 M-mode")
        self.resize(1200, 800)

        self._raw_frames:  dict[int, np.ndarray] = {}   # 原始 RF
        self._proc_frames: dict[int, np.ndarray] = {}   # 处理后 [0,1]
        self._bufs:        dict[int, np.ndarray] = {}   # 显示缓冲
        self._n_frames  = 0
        self._cur_frame = 0
        self._playing   = False
        self._history   = 300

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._img_items: dict[int, pg.ImageItem] = {}
        self._proc_thread = None

        self._build_ui()

        if csv_file:
            self._load_file(csv_file)

    # ── UI 构建 ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── 第一行：文件 / 显示帧数 / 采样点显示范围 ─────────────────────────
        row1 = QHBoxLayout()
        btn_open = QPushButton("打开 CSV"); btn_open.setFixedWidth(90)
        btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("未选择文件"); self._lbl_file.setStyleSheet("color:gray")
        row1.addWidget(btn_open)
        row1.addWidget(self._lbl_file, stretch=1)

        row1.addWidget(QLabel("显示帧数:"))
        self._spin_hist = QSpinBox()
        self._spin_hist.setRange(10, 5000); self._spin_hist.setValue(300); self._spin_hist.setFixedWidth(70)
        self._spin_hist.valueChanged.connect(self._on_hist_changed)
        row1.addWidget(self._spin_hist)

        row1.addWidget(QLabel("采样点:"))
        self._spin_s0 = QSpinBox(); self._spin_s0.setRange(0, 999);  self._spin_s0.setValue(0);    self._spin_s0.setFixedWidth(60)
        self._spin_s1 = QSpinBox(); self._spin_s1.setRange(1, 1000); self._spin_s1.setValue(1000); self._spin_s1.setFixedWidth(60)
        btn_s = QPushButton("应用"); btn_s.setFixedWidth(45); btn_s.clicked.connect(self._redraw)
        row1.addWidget(self._spin_s0); row1.addWidget(QLabel("~")); row1.addWidget(self._spin_s1); row1.addWidget(btn_s)
        root.addLayout(row1)

        # ── 第二行：处理参数 ──────────────────────────────────────────────────
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("TGC(dB/采样点):"))
        self._spin_tgc = QDoubleSpinBox()
        self._spin_tgc.setRange(0.0, 0.1); self._spin_tgc.setValue(0.0)
        self._spin_tgc.setDecimals(4); self._spin_tgc.setSingleStep(0.001); self._spin_tgc.setFixedWidth(75)
        row2.addWidget(self._spin_tgc)

        row2.addWidget(QLabel("动态范围(dB):"))
        self._spin_dr = QDoubleSpinBox()
        self._spin_dr.setRange(1.0, 60.0); self._spin_dr.setValue(15.0)
        self._spin_dr.setDecimals(1); self._spin_dr.setFixedWidth(65)
        row2.addWidget(self._spin_dr)

        row2.addWidget(QLabel("深度平滑σ:"))
        self._spin_sd = QDoubleSpinBox()
        self._spin_sd.setRange(0.0, 10.0); self._spin_sd.setValue(1.5)
        self._spin_sd.setDecimals(1); self._spin_sd.setFixedWidth(55)
        row2.addWidget(self._spin_sd)

        row2.addWidget(QLabel("时间平滑σ:"))
        self._spin_st = QDoubleSpinBox()
        self._spin_st.setRange(0.0, 10.0); self._spin_st.setValue(1.0)
        self._spin_st.setDecimals(1); self._spin_st.setFixedWidth(55)
        row2.addWidget(self._spin_st)

        self._btn_proc = QPushButton("重新处理"); self._btn_proc.setFixedWidth(80)
        self._btn_proc.setEnabled(False)
        self._btn_proc.clicked.connect(self._reprocess)
        row2.addWidget(self._btn_proc)

        self._lbl_proc = QLabel(""); self._lbl_proc.setStyleSheet("color:#aaa")
        row2.addWidget(self._lbl_proc)
        row2.addStretch()
        root.addLayout(row2)

        # ── 4通道图像区域（2×2）──────────────────────────────────────────────
        lut = _gray_lut()
        grid_top = QHBoxLayout(); grid_top.setSpacing(4)
        grid_bot = QHBoxLayout(); grid_bot.setSpacing(4)
        for ch in _CHANNELS:
            pw = pg.PlotWidget()
            pw.setLabel("left",   f"ch{ch} 采样点", **{"font-size": "9px"})
            pw.setLabel("bottom", "帧",                **{"font-size": "9px"})
            pw.getAxis("left").setWidth(48)
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

        self._raw_frames = {}
        for ch in sorted(df["channel"].unique().tolist()):
            sub = df[df["channel"] == ch].reset_index(drop=True)
            self._raw_frames[int(ch)] = sub[dcols].values.astype(np.float32)

        self._lbl_file.setText(Path(path).name)
        self._lbl_file.setStyleSheet("color:white")
        self._reprocess()

    # ── 预处理 ────────────────────────────────────────────────────────────────
    def _make_processor(self) -> UltrasoundProcessor:
        return UltrasoundProcessor(
            tgc_slope    = float(self._spin_tgc.value()),
            dynamic_range= float(self._spin_dr.value()),
            sigma_depth  = float(self._spin_sd.value()),
            sigma_time   = float(self._spin_st.value()),
        )

    def _reprocess(self):
        if not self._raw_frames:
            return
        self._lbl_proc.setText("处理中...")
        self._btn_proc.setEnabled(False)
        self._btn_play.setEnabled(False)
        self._on_stop()

        proc = self._make_processor()
        self._proc_thread = ProcessThread(self._raw_frames, proc)
        self._proc_thread.finished.connect(self._on_processed)
        self._proc_thread.start()

    def _on_processed(self, result: dict):
        self._proc_frames = result
        first = min(self._proc_frames)
        self._n_frames  = len(self._proc_frames[first])
        self._cur_frame = 0
        self._reset_bufs()
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._btn_play.setEnabled(True)
        self._btn_proc.setEnabled(True)
        self._lbl_proc.setText(f"完成  {self._n_frames} 帧")
        self._render(0)

    # ── 缓冲 & 渲染 ───────────────────────────────────────────────────────────
    def _reset_bufs(self):
        self._bufs = {ch: np.zeros((self._history, _N_SAMPLES), dtype=np.float32)
                      for ch in self._proc_frames}

    def _render_img(self, ch: int, buf: np.ndarray):
        s0 = max(0, int(self._spin_s0.value())); s1 = min(_N_SAMPLES, int(self._spin_s1.value()))
        if s1 <= s0: s1 = s0 + 1

        sl  = buf[:, s0:s1]
        img = (sl * 255).astype(np.uint8)

        item = self._img_items.get(ch)
        if item:
            item.setImage(img, autoLevels=False, levels=(0, 255))
            item.setRect(pg.QtCore.QRectF(0, s0, self._history, s1 - s0))

    def _render(self, frame_idx: int):
        if not self._proc_frames or frame_idx >= self._n_frames:
            return
        for ch, frames in self._proc_frames.items():
            buf = self._bufs[ch]
            if frame_idx < len(frames):
                buf = np.roll(buf, -1, axis=0)
                buf[-1] = frames[frame_idx]
                self._bufs[ch] = buf
            self._render_img(ch, buf)

        self._lbl_frame.setText(f"帧: {frame_idx} / {self._n_frames - 1}")
        self._slider.setValue(frame_idx)
        self._cur_frame = frame_idx

    def _redraw(self):
        if not self._proc_frames:
            return
        start = max(0, self._cur_frame - self._history + 1)
        self._reset_bufs()
        for i in range(start, self._cur_frame + 1):
            for ch, frames in self._proc_frames.items():
                if i < len(frames):
                    self._bufs[ch] = np.roll(self._bufs[ch], -1, axis=0)
                    self._bufs[ch][-1] = frames[i]
        for ch, buf in self._bufs.items():
            self._render_img(ch, buf)

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
        self._btn_play.setEnabled(bool(self._proc_frames)); self._btn_pause.setEnabled(False)
        self._cur_frame = 0; self._reset_bufs()
        if self._proc_frames: self._render(0)

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
        if self._proc_frames:
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
