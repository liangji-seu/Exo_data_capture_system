#!/usr/bin/env python3
"""
ultrasound_viewer.py — 工业级去噪增强版
功能：
  1. 2D 中值滤波 + 动态黑场压制（彻底消除雪花噪点）
  2. 逐帧自适应动态范围拉伸（保持图像亮度稳定）
  3. 边缘保护平滑 + 智能锐化管线
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace, median_filter
from scipy.signal import hilbert

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

# 基础配置
pg.setConfigOptions(antialias=False, background="k", foreground="w")
_CHANNELS = [1, 2, 3, 4]

# ─────────────────────────────────────────────────────────────────────────────
#  去噪增强处理器
# ─────────────────────────────────────────────────────────────────────────────

class UltrasoundProcessor:
    def __init__(self, tgc_slope=0.025, sigma_depth=1.2, sigma_time=0.8, sharpen=0.2):
        self.tgc_slope = tgc_slope
        self.sigma_depth = sigma_depth
        self.sigma_time = sigma_time
        self.sharpen = sharpen

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """
        输入: (N_frames, 1000) 原始 RF 信号
        输出: (N_frames, 1000) 归一化去噪图像 [0.0, 1.0]
        """
        raw_float = raw.astype(np.float32)
        n_frames, n_samples = raw_float.shape

        # 1. 包络提取 (希尔伯特变换)
        rf_centered = raw_float - np.mean(raw_float, axis=1, keepdims=True)
        envelope = np.abs(hilbert(rf_centered, axis=1))
        
        # 2. 对数压缩
        log_env = 20.0 * np.log10(np.clip(envelope, 1e-6, None))

        # 3. 逐帧自适应优化 (Adaptive AGC)
        processed = np.zeros_like(log_env)
        tgc_gain = np.arange(n_samples) * self.tgc_slope
        
        for i in range(n_frames):
            frame = log_env[i] + tgc_gain
            
            # --- 关键：提高黑场阈值到 25%，强力压制背景噪声 ---
            vmin = np.percentile(frame, 25.0) 
            vmax = np.percentile(frame, 99.0)
            
            processed[i] = np.clip((frame - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)

        # 4. 强力去噪 (3x3 2D 中值滤波)
        # 这一步能精准干掉图片里的“白雪花”
        processed = median_filter(processed, size=(3, 3)) 

        # 5. 2D 高斯平滑 (构建丝滑感)
        if self.sigma_depth > 0 or self.sigma_time > 0:
            processed = gaussian_filter(processed, sigma=[self.sigma_time, self.sigma_depth])

        # 6. 最终锐化 (只增强主要肌肉轮廓)
        if self.sharpen > 0:
            edges = laplace(processed)
            processed = np.clip(processed - self.sharpen * edges, 0.0, 1.0)

        return processed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  后台处理线程
# ─────────────────────────────────────────────────────────────────────────────

class ProcessThread(QThread):
    finished = Signal(dict)

    def __init__(self, raw_frames: dict, processor: UltrasoundProcessor):
        super().__init__()
        self._raw = raw_frames
        self._proc = processor

    def run(self):
        result = {}
        for ch, frames in self._raw.items():
            result[ch] = self._proc.preprocess(frames)
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
#  UI 界面逻辑
# ─────────────────────────────────────────────────────────────────────────────

def _gray_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
    return lut

class UltrasoundViewer(QWidget):
    def __init__(self, csv_file=""):
        super().__init__()
        self.setWindowTitle("RF-Ultrasound Pro (去噪增强版)")
        self.resize(1280, 850)

        self._raw_frames = {}   
        self._proc_frames = {}   
        self._bufs = {}   
        self._n_frames = 0
        self._cur_frame = 0
        self._playing = False
        self._history = 400 
        self._n_samples_max = 1000

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._img_items = {}
        self._build_ui()

        if csv_file:
            self._load_file(csv_file)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # 顶部工具栏
        top = QHBoxLayout()
        btn_open = QPushButton("打开数据 (CSV)")
        btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("待载入...")
        top.addWidget(btn_open)
        top.addWidget(self._lbl_file, stretch=1)

        top.addWidget(QLabel("时间轴(帧):"))
        self._spin_hist = QSpinBox()
        self._spin_hist.setRange(50, 2000); self._spin_hist.setValue(400)
        self._spin_hist.valueChanged.connect(self._on_hist_changed)
        top.addWidget(self._spin_hist)

        top.addWidget(QLabel("视野范围:"))
        self._spin_s0 = QSpinBox(); self._spin_s0.setRange(0, 1000); self._spin_s0.setValue(150)
        self._spin_s1 = QSpinBox(); self._spin_s1.setRange(0, 1000); self._spin_s1.setValue(850)
        top.addWidget(self._spin_s0); top.addWidget(QLabel("-")); top.addWidget(self._spin_s1)
        btn_apply = QPushButton("应用视野"); btn_apply.clicked.connect(self._redraw)
        top.addWidget(btn_apply)
        root.addLayout(top)

        # 参数调节区
        param = QHBoxLayout()
        param.addWidget(QLabel("深层补偿(TGC):"))
        self._spin_tgc = QDoubleSpinBox()
        self._spin_tgc.setRange(0, 0.2); self._spin_tgc.setValue(0.025); self._spin_tgc.setDecimals(3)
        param.addWidget(self._spin_tgc)

        param.addWidget(QLabel("锐化强度:"))
        self._spin_shp = QDoubleSpinBox()
        self._spin_shp.setRange(0, 1.0); self._spin_shp.setValue(0.20)
        param.addWidget(self._spin_shp)

        param.addWidget(QLabel("降噪强度(σ):"))
        self._spin_sd = QDoubleSpinBox()
        self._spin_sd.setRange(0, 5); self._spin_sd.setValue(1.2)
        param.addWidget(self._spin_sd)

        self._btn_reproc = QPushButton("⚡ 刷新增强图像")
        self._btn_reproc.setStyleSheet("font-weight: bold; color: #00ff00; background: #222")
        self._btn_reproc.clicked.connect(self._reprocess)
        param.addWidget(self._btn_reproc)
        param.addStretch()
        root.addLayout(param)

        # 图表显示区
        lut = _gray_lut()
        grid = QVBoxLayout()
        for i in range(2): 
            row_ly = QHBoxLayout()
            for j in range(2):
                ch = i * 2 + j + 1
                pw = pg.PlotWidget(title=f"Channel {ch}")
                pw.invertY(True)
                img = pg.ImageItem()
                img.setLookupTable(lut)
                pw.addItem(img)
                self._img_items[ch] = img
                row_ly.addWidget(pw)
            grid.addLayout(row_ly)
        root.addLayout(grid)

        # 播放控制
        bot = QHBoxLayout()
        self._btn_play = QPushButton("播放"); self._btn_play.clicked.connect(self._on_play)
        self._btn_pause = QPushButton("暂停"); self._btn_pause.clicked.connect(self._on_pause)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.sliderMoved.connect(self._on_slider)
        self._lbl_frame = QLabel("帧: 0/0")
        
        bot.addWidget(self._btn_play); bot.addWidget(self._btn_pause)
        bot.addWidget(self._lbl_frame)
        bot.addWidget(self._slider, stretch=1)
        root.addLayout(bot)

    def _load_file(self, path):
        df = pd.read_csv(path)
        dcols = [c for c in df.columns if c.startswith('d') and c[1:].isdigit()]
        self._raw_frames = {int(ch): df[df['channel']==ch][dcols].values.astype(np.float32) 
                           for ch in df['channel'].unique()}
        self._n_samples_max = len(dcols)
        self._lbl_file.setText(Path(path).name)
        self._reprocess()

    def _reprocess(self):
        if not self._raw_frames: return
        self._btn_reproc.setText("正在去噪...")
        proc = UltrasoundProcessor(self._spin_tgc.value(), self._spin_sd.value(), 0.5, self._spin_shp.value())
        self._thread = ProcessThread(self._raw_frames, proc)
        self._thread.finished.connect(self._on_done)
        self._thread.start()

    def _on_done(self, result):
        self._proc_frames = result
        self._n_frames = len(next(iter(result.values())))
        self._slider.setMaximum(self._n_frames - 1)
        self._reset_bufs()
        self._btn_reproc.setText("⚡ 刷新增强图像")
        self._render(0)

    def _reset_bufs(self):
        self._bufs = {ch: np.zeros((self._history, self._n_samples_max)) for ch in self._proc_frames}

    def _render(self, idx):
        if not self._proc_frames: return
        s0, s1 = self._spin_s0.value(), self._spin_s1.value()
        for ch, frames in self._proc_frames.items():
            self._bufs[ch] = np.roll(self._bufs[ch], -1, axis=0)
            self._bufs[ch][-1] = frames[idx]
            img_data = (self._bufs[ch][:, s0:s1] * 255).astype(np.uint8)
            self._img_items[ch].setImage(img_data, autoLevels=False, levels=(0, 255))
            self._img_items[ch].setRect(pg.QtCore.QRectF(0, s0, self._history, s1-s0))
        self._lbl_frame.setText(f"帧: {idx}/{self._n_frames}")
        self._slider.setValue(idx)
        self._cur_frame = idx

    def _next_frame(self):
        if self._cur_frame < self._n_frames - 1: self._render(self._cur_frame + 1)
        else: self._on_pause()

    def _on_play(self): self._play_timer.start(30); self._playing = True
    def _on_pause(self): self._play_timer.stop(); self._playing = False
    def _on_slider(self, v): self._on_pause(); self._reset_bufs(); self._render(v)
    def _on_hist_changed(self, v): self._history = v; self._reset_bufs(); self._redraw()
    def _redraw(self): self._render(self._cur_frame)
    def _on_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 CSV", "", "CSV (*.csv)")
        if p: self._load_file(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = UltrasoundViewer()
    view.show()
    sys.exit(app.exec())