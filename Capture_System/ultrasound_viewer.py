#!/usr/bin/env python3
"""
ultrasound_viewer.py — 工业级去噪增强版 + 力矩同步显示
功能：
  1. 2D 中值滤波 + 动态黑场压制
  2. 逐帧自适应动态范围拉伸
  3. 边缘保护平滑 + 智能锐化管线
  4. 可加载 torque.csv，与超声帧时间戳同步显示步态相位和髋关节力矩
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

pg.setConfigOptions(antialias=False, background="k", foreground="w")
_CHANNELS = [1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
#  去噪增强处理器
# ─────────────────────────────────────────────────────────────────────────────

class UltrasoundProcessor:
    def __init__(self, tgc_slope=0.025, sigma_depth=1.2, sigma_time=0.8, sharpen=0.2):
        self.tgc_slope   = tgc_slope
        self.sigma_depth = sigma_depth
        self.sigma_time  = sigma_time
        self.sharpen     = sharpen

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        raw_float = raw.astype(np.float32)
        n_frames, n_samples = raw_float.shape

        rf_centered = raw_float - np.mean(raw_float, axis=1, keepdims=True)
        envelope    = np.abs(hilbert(rf_centered, axis=1))
        log_env     = 20.0 * np.log10(np.clip(envelope, 1e-6, None))

        processed = np.zeros_like(log_env)
        tgc_gain  = np.arange(n_samples) * self.tgc_slope
        for i in range(n_frames):
            frame = log_env[i] + tgc_gain
            vmin  = np.percentile(frame, 25.0)
            vmax  = np.percentile(frame, 99.0)
            processed[i] = np.clip((frame - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)

        processed = median_filter(processed, size=(3, 3))
        if self.sigma_depth > 0 or self.sigma_time > 0:
            processed = gaussian_filter(processed, sigma=[self.sigma_time, self.sigma_depth])
        if self.sharpen > 0:
            edges     = laplace(processed)
            processed = np.clip(processed - self.sharpen * edges, 0.0, 1.0)

        return processed.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  后台处理线程
# ─────────────────────────────────────────────────────────────────────────────

class ProcessThread(QThread):
    finished = Signal(dict)

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
    def __init__(self, csv_file=""):
        super().__init__()
        self.setWindowTitle("RF-Ultrasound Pro (去噪增强版)")
        self.resize(1280, 950)

        self._raw_frames:  dict = {}
        self._proc_frames: dict = {}
        self._bufs:        dict = {}
        self._ult_times:   np.ndarray = np.array([])  # 超声帧时间戳
        self._n_frames     = 0
        self._cur_frame    = 0
        self._playing      = False
        self._history      = 400
        self._n_samples_max = 1000

        # torque 数据
        self._torque_ts:     np.ndarray = np.array([])
        self._torque_phase:  np.ndarray = np.array([])
        self._torque_torque: np.ndarray = np.array([])
        # 滚动缓冲（与超声帧同步，长度跟随 _history）
        self._phase_buf:  np.ndarray = np.zeros(400)
        self._torque_buf: np.ndarray = np.zeros(400)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._img_items = {}
        self._build_ui()

        if csv_file:
            self._load_file(csv_file)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # 第一行：文件操作
        top = QHBoxLayout()
        btn_open = QPushButton("打开超声 CSV"); btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("待载入...")
        top.addWidget(btn_open); top.addWidget(self._lbl_file, stretch=1)

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

        # 第二行：处理参数 + 加载 torque
        param = QHBoxLayout()
        param.addWidget(QLabel("TGC:"))
        self._spin_tgc = QDoubleSpinBox()
        self._spin_tgc.setRange(0, 0.2); self._spin_tgc.setValue(0.025); self._spin_tgc.setDecimals(3)
        param.addWidget(self._spin_tgc)

        param.addWidget(QLabel("锐化:"))
        self._spin_shp = QDoubleSpinBox()
        self._spin_shp.setRange(0, 1.0); self._spin_shp.setValue(0.20)
        param.addWidget(self._spin_shp)

        param.addWidget(QLabel("降噪σ:"))
        self._spin_sd = QDoubleSpinBox()
        self._spin_sd.setRange(0, 5); self._spin_sd.setValue(1.2)
        param.addWidget(self._spin_sd)

        self._btn_reproc = QPushButton("⚡ 刷新增强图像")
        self._btn_reproc.setStyleSheet("font-weight:bold; color:#00ff00; background:#222")
        self._btn_reproc.clicked.connect(self._reprocess)
        param.addWidget(self._btn_reproc)

        param.addSpacing(20)
        btn_torque = QPushButton("加载 torque.csv"); btn_torque.clicked.connect(self._on_open_torque)
        self._lbl_torque = QLabel("未加载"); self._lbl_torque.setStyleSheet("color:#888")
        param.addWidget(btn_torque); param.addWidget(self._lbl_torque)
        param.addStretch()
        root.addLayout(param)

        # 超声图像区（2×2），每个通道叠加力矩/相位曲线
        lut  = _gray_lut()
        grid = QVBoxLayout(); grid.setSpacing(4)
        self._plot_widgets:  dict[int, pg.PlotWidget]    = {}
        self._phase_curves:  dict[int, pg.PlotDataItem]  = {}
        self._torque_curves: dict[int, pg.PlotDataItem]  = {}

        for i in range(2):
            row_ly = QHBoxLayout(); row_ly.setSpacing(4)
            for j in range(2):
                ch = i * 2 + j + 1
                pw = pg.PlotWidget(title=f"Channel {ch}")
                pw.invertY(True)
                pw.getAxis("bottom").setHeight(18)
                pw.getAxis("left").setWidth(38)

                img = pg.ImageItem()
                img.setLookupTable(lut)
                img.setZValue(0)
                pw.addItem(img)
                self._img_items[ch]    = img
                self._plot_widgets[ch] = pw

                # 曲线直接加到主 PlotItem，z 值高于 ImageItem，保证显示在上层
                c_phase  = pw.plot(pen=pg.mkPen("#FF5722", width=2))
                c_torque = pw.plot(pen=pg.mkPen("#4CAF50", width=2))
                c_phase.setZValue(10)
                c_torque.setZValue(10)
                self._phase_curves[ch]  = c_phase
                self._torque_curves[ch] = c_torque

                # 步态周期竖线（红色，贯穿整个 y 轴）
                c_cycle = pw.plot(pen=pg.mkPen("#FF0000", width=1.5))
                c_cycle.setZValue(11)
                if not hasattr(self, '_cycle_lines'):
                    self._cycle_lines = {}
                self._cycle_lines[ch] = c_cycle

                row_ly.addWidget(pw)
            grid.addLayout(row_ly)
        root.addLayout(grid, stretch=1)

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
        p, _ = QFileDialog.getOpenFileName(self, "选择超声 CSV", "", "CSV (*.csv)")
        if p:
            self._load_file(p)

    def _on_open_torque(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 torque.csv", "", "CSV (*.csv)")
        if p:
            self._load_torque(p)

    def _load_file(self, path):
        df    = pd.read_csv(path)
        dcols = [c for c in df.columns if c.startswith('d') and c[1:].isdigit()]
        self._raw_frames = {}
        self._ult_times  = np.array([])

        for ch in sorted(df['channel'].unique()):
            sub = df[df['channel'] == ch].reset_index(drop=True)
            self._raw_frames[int(ch)] = sub[dcols].values.astype(np.float32)
            if self._ult_times.size == 0 and 'timestamp' in sub.columns:
                self._ult_times = sub['timestamp'].values.astype(np.float64)

        self._n_samples_max = len(dcols)
        self._lbl_file.setText(Path(path).name)
        self._reprocess()

        # 尝试自动加载同目录的 torque.csv
        auto = Path(path).parent / "torque.csv"
        if auto.exists() and self._torque_ts.size == 0:
            self._load_torque(str(auto))

    def _load_torque(self, path: str):
        try:
            df = pd.read_csv(path)
            if "timestamp" not in df.columns:
                self._lbl_torque.setText("格式错误"); return
            self._torque_ts     = df["timestamp"].values.astype(np.float64)
            self._torque_phase  = df["phase_pct"].values.astype(np.float64) if "phase_pct" in df.columns else np.zeros(len(df))
            self._torque_torque = df["hip_torque_Nm"].values.astype(np.float64) if "hip_torque_Nm" in df.columns else np.zeros(len(df))
            self._lbl_torque.setText(f"{Path(path).name}  ({len(df)} 帧)")
            self._lbl_torque.setStyleSheet("color:#00cc00")
            self._reset_torque_bufs()
        except Exception as e:
            self._lbl_torque.setText(f"加载失败: {e}")

    # ── 预处理 ────────────────────────────────────────────────────────────
    def _reprocess(self):
        if not self._raw_frames: return
        self._btn_reproc.setText("正在去噪...")
        proc = UltrasoundProcessor(self._spin_tgc.value(), self._spin_sd.value(), 0.5, self._spin_shp.value())
        self._thread = ProcessThread(self._raw_frames, proc)
        self._thread.finished.connect(self._on_done)
        self._thread.start()

    def _on_done(self, result):
        self._proc_frames = result
        self._n_frames    = len(next(iter(result.values())))
        self._slider.setMaximum(self._n_frames - 1)
        self._reset_bufs()
        self._btn_reproc.setText("⚡ 刷新增强图像")
        self._render(0)

    # ── 缓冲 & 渲染 ───────────────────────────────────────────────────────
    def _reset_bufs(self):
        self._bufs = {ch: np.zeros((self._history, self._n_samples_max))
                      for ch in self._proc_frames}
        self._reset_torque_bufs()

    def _reset_torque_bufs(self):
        self._phase_buf  = np.zeros(self._history)
        self._torque_buf = np.zeros(self._history)
        # 清除各通道的步态周期竖线
        for ch in list(getattr(self, '_cycle_lines', {}).keys()):
            self._cycle_lines[ch].setData([], [])
        if not hasattr(self, '_cycle_lines'):
            self._cycle_lines = {}

    def _lookup_torque(self, ts: float):
        """根据时间戳找最近的 torque/phase 值，找不到返回 (0, 0)。"""
        if self._torque_ts.size == 0:
            return 0.0, 0.0
        idx = int(np.argmin(np.abs(self._torque_ts - ts)))
        return float(self._torque_phase[idx]), float(self._torque_torque[idx])

    def _render(self, idx):
        if not self._proc_frames: return
        s0, s1 = self._spin_s0.value(), self._spin_s1.value()
        x_axis = np.arange(self._history)

        # 查找当前帧对应的力矩/相位值，推入滚动缓冲
        if self._ult_times.size > idx:
            phase_val, torque_val = self._lookup_torque(float(self._ult_times[idx]))
        else:
            phase_val, torque_val = 0.0, 0.0
        self._phase_buf  = np.roll(self._phase_buf,  -1); self._phase_buf[-1]  = phase_val
        self._torque_buf = np.roll(self._torque_buf, -1); self._torque_buf[-1] = torque_val

        # 检测步态周期结束（相位从高值回落到低值 = 下降沿）
        # 用 nan 分隔的竖线段：每条竖线 = [x, x, nan], [y_min, y_max, nan]
        cycle_xs = []
        cycle_ys = []
        if self._torque_ts.size > 0 and len(self._phase_buf) > 1:
            for i in range(1, len(self._phase_buf)):
                if self._phase_buf[i - 1] > 90 and self._phase_buf[i] < 10:
                    cycle_xs += [i, i, float('nan')]
                    cycle_ys += [s0, s1, float('nan')]

        for ch, frames in self._proc_frames.items():
            # 超声图像
            self._bufs[ch] = np.roll(self._bufs[ch], -1, axis=0)
            self._bufs[ch][-1] = frames[idx]
            img_data = (self._bufs[ch][:, s0:s1] * 255).astype(np.uint8)
            self._img_items[ch].setImage(img_data, autoLevels=False, levels=(0, 255))
            self._img_items[ch].setRect(pg.QtCore.QRectF(0, s0, self._history, s1 - s0))

            # 叠加力矩/相位曲线（同一 x 轴，从右往左滚动）
            if self._torque_ts.size > 0:
                self._phase_curves[ch].setData(x_axis, self._phase_buf)
                self._torque_curves[ch].setData(x_axis, self._torque_buf)
                # 步态周期竖线
                if cycle_xs:
                    self._cycle_lines[ch].setData(cycle_xs, cycle_ys)
                else:
                    self._cycle_lines[ch].setData([], [])

        self._lbl_frame.setText(f"帧: {idx}/{self._n_frames}")
        self._slider.setValue(idx)
        self._cur_frame = idx

    def _redraw(self):
        self._render(self._cur_frame)

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
        self._redraw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="", help="超声 CSV 路径")
    args = parser.parse_args()
    app  = QApplication(sys.argv)
    win  = UltrasoundViewer(csv_file=args.file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
