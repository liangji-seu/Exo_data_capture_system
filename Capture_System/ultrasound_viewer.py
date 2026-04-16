"""
超声灰度图播放器
读取 capture_ultrasound.py 生成的 CSV 文件，
将 4 个通道的波形数据以灰度图（B超扫描线堆叠）形式显示，并支持时间轴播放。

CSV 格式：
    timestamp, channel, pack_num, d0, d1, ..., d999

运行：
    python ultrasound_viewer.py
    python ultrasound_viewer.py --file ./data/20260416_150112/ultrasound_20260416_150112.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore import QThread, QTimer, Signal, Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog, QGroupBox,
        QSpinBox, QDoubleSpinBox,
    )
except ImportError:
    from PyQt5.QtCore import QThread, QTimer, pyqtSignal as Signal, Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog, QGroupBox,
        QSpinBox, QDoubleSpinBox,
    )

import pyqtgraph as pg


# ─────────────────────────────────────────────────────────────────────────────
#  CSV 加载线程（防止大文件卡 UI）
# ─────────────────────────────────────────────────────────────────────────────

class LoadThread(QThread):
    finished = Signal(object, str)   # (result_dict or None, error_msg)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def run(self):
        try:
            df = pd.read_csv(self._path)
            data_cols = [c for c in df.columns if c.startswith("d")]
            channels = sorted(df["channel"].unique().tolist())

            # 按通道分组，每通道得到 (N帧 × 1000点) 的矩阵和对应时间戳
            ch_frames = {}
            ch_times  = {}
            for ch in channels:
                sub = df[df["channel"] == ch].reset_index(drop=True)
                ch_frames[int(ch)] = sub[data_cols].values.astype(np.float32)
                ch_times[int(ch)]  = sub["timestamp"].values.astype(np.float64)

            self.finished.emit({"frames": ch_frames, "times": ch_times,
                                 "channels": [int(c) for c in channels]}, "")
        except Exception as e:
            self.finished.emit(None, str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────

class UltrasoundViewer(QWidget):
    def __init__(self, init_file: str = ""):
        super().__init__()
        self._data       = None    # {"frames": {ch: ndarray}, "times": {ch: ndarray}}
        self._channels   = []
        self._n_frames   = 0       # 以通道1（或第一个通道）的帧数为基准
        self._cur_frame  = 0
        self._playing    = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)
        self._load_thread = None

        self._img_items: dict[int, pg.ImageItem] = {}   # {ch: ImageItem}
        self._history_len = 200   # 灰度图纵轴：显示最近多少帧

        self._build_ui()

        if init_file:
            self._load_file(init_file)

    # ── 界面构建 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("超声灰度图播放器")
        self.resize(1100, 720)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel("超声灰度图播放器")
        title.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(13); f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        # ── 文件选择行 ────────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self._lbl_file = QLabel("未加载文件")
        self._lbl_file.setStyleSheet("color: gray;")
        btn_open = QPushButton("选择 CSV 文件")
        btn_open.setFixedWidth(130)
        btn_open.clicked.connect(self._on_open)
        file_row.addWidget(btn_open)
        file_row.addWidget(self._lbl_file, stretch=1)
        root.addLayout(file_row)

        # ── 状态标签 ──────────────────────────────────────────────────────
        self._lbl_status = QLabel("请先加载 CSV 文件")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._lbl_status)

        # ── 灰度图区域（4 通道 2×2）────────────────────────────────────────
        # 横轴 = 采样点（0~999），纵轴 = 帧序/时间
        pg.setConfigOptions(antialias=False, background="k", foreground="w")
        img_group = QGroupBox("灰度图（横轴=采样点，纵轴=帧序/时间）")
        img_layout = QVBoxLayout(img_group); img_layout.setSpacing(4)
        row1 = QHBoxLayout(); row1.setSpacing(4)
        row2 = QHBoxLayout(); row2.setSpacing(4)

        self._plot_widgets: dict[int, pg.PlotWidget] = {}
        for ch in (1, 2, 3, 4):
            pw = pg.PlotWidget(title=f"通道 {ch}")
            pw.setLabel("bottom", "采样点")
            pw.setLabel("left", "帧")
            pw.invertY(False)
            img_item = pg.ImageItem()
            img_item.setLookupTable(self._gray_lut())
            pw.addItem(img_item)
            self._img_items[ch] = img_item
            self._plot_widgets[ch] = pw
            (row1 if ch <= 2 else row2).addWidget(pw)

        img_layout.addLayout(row1)
        img_layout.addLayout(row2)
        root.addWidget(img_group, stretch=4)

        # ── 播放控制行 ────────────────────────────────────────────────────
        ctrl_group = QGroupBox("播放控制")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # 进度滑条
        slider_row = QHBoxLayout()
        self._lbl_frame = QLabel("帧: 0 / 0")
        self._lbl_frame.setFixedWidth(110)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._lbl_time = QLabel("时间: 0.000 s")
        self._lbl_time.setFixedWidth(130)
        slider_row.addWidget(self._lbl_frame)
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._lbl_time)
        ctrl_layout.addLayout(slider_row)

        # 按钮行
        btn_row = QHBoxLayout()
        self._btn_play  = QPushButton("▶ 播放")
        self._btn_pause = QPushButton("⏸ 暂停")
        self._btn_stop  = QPushButton("⏹ 停止")
        self._btn_pause.setEnabled(False)
        for b in (self._btn_play, self._btn_pause, self._btn_stop):
            b.setFixedHeight(32)
            b.setFixedWidth(90)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        # 播放速度
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(0.1, 200.0)
        self._spin_fps.setValue(10.0)
        self._spin_fps.setDecimals(1)
        self._spin_fps.setFixedWidth(75)
        self._spin_fps.setSuffix(" fps")
        self._spin_fps.valueChanged.connect(self._on_fps_changed)

        # 显示历史帧数
        self._spin_history = QSpinBox()
        self._spin_history.setRange(10, 2000)
        self._spin_history.setValue(self._history_len)
        self._spin_history.setFixedWidth(75)
        self._spin_history.valueChanged.connect(self._on_history_changed)

        # 灰度值范围
        self._spin_vmin = QDoubleSpinBox()
        self._spin_vmin.setRange(0, 1e6)
        self._spin_vmin.setValue(0)
        self._spin_vmin.setDecimals(0)
        self._spin_vmin.setFixedWidth(80)

        self._spin_vmax = QDoubleSpinBox()
        self._spin_vmax.setRange(0, 1e6)
        self._spin_vmax.setValue(300)
        self._spin_vmax.setDecimals(0)
        self._spin_vmax.setFixedWidth(80)

        btn_apply_v = QPushButton("应用")
        btn_apply_v.setFixedWidth(50)
        btn_apply_v.setFixedHeight(28)
        btn_apply_v.clicked.connect(self._apply_levels)

        # 采样点显示范围
        self._spin_smin = QSpinBox()
        self._spin_smin.setRange(0, 999)
        self._spin_smin.setValue(0)
        self._spin_smin.setFixedWidth(70)

        self._spin_smax = QSpinBox()
        self._spin_smax.setRange(1, 1000)
        self._spin_smax.setValue(1000)
        self._spin_smax.setFixedWidth(70)

        btn_apply_s = QPushButton("应用")
        btn_apply_s.setFixedWidth(50)
        btn_apply_s.setFixedHeight(28)
        btn_apply_s.clicked.connect(self._apply_sample_range)

        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_pause)
        btn_row.addWidget(self._btn_stop)
        btn_row.addSpacing(20)
        btn_row.addWidget(QLabel("播放速度:"))
        btn_row.addWidget(self._spin_fps)
        btn_row.addSpacing(20)
        btn_row.addWidget(QLabel("显示帧数:"))
        btn_row.addWidget(self._spin_history)
        btn_row.addSpacing(20)
        btn_row.addWidget(QLabel("采样点范围:"))
        btn_row.addWidget(self._spin_smin)
        btn_row.addWidget(QLabel("~"))
        btn_row.addWidget(self._spin_smax)
        btn_row.addWidget(btn_apply_s)
        btn_row.addSpacing(20)
        btn_row.addWidget(QLabel("灰度范围:"))
        btn_row.addWidget(self._spin_vmin)
        btn_row.addWidget(QLabel("~"))
        btn_row.addWidget(self._spin_vmax)
        btn_row.addWidget(btn_apply_v)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)
        root.addWidget(ctrl_group)

    # ── 灰度 LUT（黑→白）────────────────────────────────────────────────────
    @staticmethod
    def _gray_lut():
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 0] = np.arange(256)
        lut[:, 1] = np.arange(256)
        lut[:, 2] = np.arange(256)
        return lut

    # ── 文件选择 ──────────────────────────────────────────────────────────────
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择超声 CSV 文件", str(Path.cwd()), "CSV 文件 (*.csv)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        self._on_stop()
        self._lbl_file.setText(Path(path).name)
        self._lbl_file.setStyleSheet("color: white;")
        self._lbl_status.setText("正在加载...")
        self._load_thread = LoadThread(path)
        self._load_thread.finished.connect(self._on_loaded)
        self._load_thread.start()

    def _on_loaded(self, result, err):
        if result is None:
            self._lbl_status.setText(f"加载失败: {err}")
            return

        self._data     = result
        self._channels = result["channels"]
        # 以第一个通道帧数为准（各通道帧数应相近）
        first_ch = self._channels[0]
        self._n_frames = len(result["times"][first_ch])

        self._cur_frame = 0
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)

        # 初始化灰度缓冲（每通道 history_len × 1000 的滚动窗口）
        self._init_buffers()
        self._render_frame(0)

        self._lbl_status.setText(
            f"已加载  通道: {self._channels}  帧数: {self._n_frames}"
        )
        self._btn_play.setEnabled(True)

    # ── 灰度缓冲初始化 ────────────────────────────────────────────────────────
    def _init_buffers(self):
        """每个通道维护一个 (history_len × 1000) 的滚动缓冲，最新帧在底部。"""
        self._bufs: dict[int, np.ndarray] = {}
        for ch in self._channels:
            self._bufs[ch] = np.zeros(
                (self._history_len, 1000), dtype=np.float32
            )

    # ── 渲染指定帧 ────────────────────────────────────────────────────────────
    def _render_frame(self, frame_idx: int):
        if self._data is None:
            return

        vmin = self._spin_vmin.value()
        vmax = self._spin_vmax.value()
        if vmin >= vmax:
            vmax = vmin + 1

        # 采样点显示范围（横轴）
        s0 = int(self._spin_smin.value())
        s1 = int(self._spin_smax.value())
        s0 = max(0, min(s0, 999))
        s1 = max(s0 + 1, min(s1, 1000))

        for ch in self._channels:
            frames = self._data["frames"][ch]
            n = len(frames)
            if frame_idx >= n:
                continue

            # 滚动：将新帧追加到缓冲末尾
            self._bufs[ch] = np.roll(self._bufs[ch], -1, axis=0)
            self._bufs[ch][-1] = frames[frame_idx]

            # 裁切采样点范围，shape: (history_len, n_samples)
            buf_slice = self._bufs[ch][:, s0:s1]

            # 归一化到 0~255
            img = np.clip((buf_slice - vmin) / (vmax - vmin), 0, 1)
            img = (img * 255).astype(np.uint8)

            # ImageItem.setImage 期望 shape=(width, height)
            # 横轴=采样点(x)，纵轴=帧(y)
            # → img.T: shape=(n_samples, history_len)，width=n_samples, height=history_len
            self._img_items[ch].setImage(img.T, autoLevels=False,
                                          levels=(0, 255))

            # 使坐标轴刻度与实际采样点编号对齐
            self._img_items[ch].setRect(
                pg.QtCore.QRectF(s0, 0, s1 - s0, self._history_len)
            )

        # 更新时间标签
        first_ch = self._channels[0]
        times = self._data["times"][first_ch]
        if frame_idx < len(times):
            t0 = times[0]
            t  = times[frame_idx]
            self._lbl_time.setText(f"时间: {t - t0:.3f} s")

        self._lbl_frame.setText(f"帧: {frame_idx} / {self._n_frames - 1}")
        self._slider.setValue(frame_idx)
        self._cur_frame = frame_idx

    # ── 播放控制 ──────────────────────────────────────────────────────────────
    def _on_play(self):
        if self._data is None:
            return
        if self._cur_frame >= self._n_frames - 1:
            self._cur_frame = 0
            self._init_buffers()
        self._playing = True
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        interval_ms = max(1, int(1000 / self._spin_fps.value()))
        self._play_timer.start(interval_ms)

    def _on_pause(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)

    def _on_stop(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setEnabled(self._data is not None)
        self._btn_pause.setEnabled(False)
        if self._data is not None:
            self._cur_frame = 0
            self._init_buffers()
            self._render_frame(0)

    def _next_frame(self):
        if self._cur_frame >= self._n_frames - 1:
            self._on_stop()
            return
        self._render_frame(self._cur_frame + 1)

    def _on_slider_moved(self, value: int):
        self._on_pause()
        self._init_buffers()
        # 从头到 value 快速填充缓冲（只填最后 history_len 帧）
        start = max(0, value - self._history_len + 1)
        for i in range(start, value + 1):
            self._render_frame(i)

    def _on_fps_changed(self, val: float):
        if self._playing:
            self._play_timer.setInterval(max(1, int(1000 / val)))

    def _on_history_changed(self, val: int):
        self._history_len = val
        if self._data is not None:
            self._init_buffers()
            self._render_frame(self._cur_frame)

    def _apply_levels(self):
        """重新渲染当前帧以应用新的灰度范围。"""
        self._render_frame(self._cur_frame)

    def _apply_sample_range(self):
        """重置缓冲并重新渲染以应用新的采样点显示范围。"""
        if self._data is not None:
            self._init_buffers()
            self._render_frame(self._cur_frame)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="超声灰度图播放器")
    parser.add_argument("--file", type=str, default="",
                        help="直接指定 CSV 文件路径")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = UltrasoundViewer(init_file=args.file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
