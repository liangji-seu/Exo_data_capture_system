"""
UI_viewer.py — 超声灰度图 + IMU 动捕数据同步回放
读取 capture_ultrasound.py 和 capture_imu.py 生成的 CSV，
用统一时间轴同步回放：
  · 超声：每通道一个灰度图（横轴=采样点，纵轴=帧）
  · IMU：每设备一个面板，上图=3轴加速度，下图=3轴角速度

用法：
    python UI_viewer.py
    python UI_viewer.py --ult ultrasound_xxx.csv --imu imu_xxx.csv
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
        QSpinBox, QDoubleSpinBox, QSplitter,
    )
except ImportError:
    from PyQt5.QtCore import QThread, QTimer, pyqtSignal as Signal, Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog, QGroupBox,
        QSpinBox, QDoubleSpinBox, QSplitter,
    )

import pyqtgraph as pg


# ─────────────────────────────────────────────────────────────────────────────
#  数据加载线程（超声 + IMU 可分别独立加载）
# ─────────────────────────────────────────────────────────────────────────────

class LoadThread(QThread):
    finished = Signal(object, str)   # (result_dict or None, errmsg)

    def __init__(self, ult_path: str = "", imu_path: str = ""):
        super().__init__()
        self._ult = ult_path
        self._imu = imu_path

    def run(self):
        res = {}
        try:
            # ── 超声 ──────────────────────────────────────────────────────
            if self._ult:
                df    = pd.read_csv(self._ult)
                dcols = [c for c in df.columns if c.startswith("d")]
                chs   = sorted(df["channel"].unique().tolist())
                frames, times = {}, {}
                for ch in chs:
                    s = df[df["channel"] == ch].reset_index(drop=True)
                    frames[int(ch)] = s[dcols].values.astype(np.float32)
                    times[int(ch)]  = s["timestamp"].values.astype(np.float64)
                res["ult"] = {"frames": frames, "times": times,
                              "channels": [int(c) for c in chs]}

            # ── IMU ───────────────────────────────────────────────────────
            if self._imu:
                df   = pd.read_csv(self._imu)
                devs = sorted(df["device_id"].unique().tolist())

                def _col(sub, name):
                    return sub[name].values.astype(np.float32) if name in sub.columns \
                           else np.zeros(len(sub), np.float32)

                dev_data = {}
                for dev in devs:
                    s = df[df["device_id"] == dev].reset_index(drop=True)
                    dev_data[str(dev)] = {
                        "times":  s["timestamp"].values.astype(np.float64),
                        "acc_x":  _col(s, "acc_x"),
                        "acc_y":  _col(s, "acc_y"),
                        "acc_z":  _col(s, "acc_z"),
                        "gyr_x":  _col(s, "gyr_x"),
                        "gyr_y":  _col(s, "gyr_y"),
                        "gyr_z":  _col(s, "gyr_z"),
                    }
                res["imu"] = {"devices": [str(d) for d in devs], "data": dev_data}

            self.finished.emit(res, "")
        except Exception as e:
            self.finished.emit(None, str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  IMU 设备滚动折线图面板
# ─────────────────────────────────────────────────────────────────────────────

class IMUDevicePanel(QGroupBox):
    """
    显示单个 IMU 设备的 3 轴加速度（上图）和 3 轴角速度（下图）滚动曲线。
    横轴为滑动窗口内的采样点编号。
    """

    def __init__(self, device_id: str, history: int = 400):
        super().__init__(f"IMU  —  {device_id}")
        self._hist = history

        ly = QVBoxLayout(self)
        ly.setContentsMargins(4, 14, 4, 4)
        ly.setSpacing(4)

        # ── 加速度图 ──────────────────────────────────────────────────────
        self._pw_acc = pg.PlotWidget(title="加速度 (m/s²)")
        self._pw_acc.setLabel("left",   "m/s²")
        self._pw_acc.setLabel("bottom", "采样点")
        self._pw_acc.showGrid(x=False, y=True, alpha=0.25)
        self._pw_acc.addLegend(offset=(5, 5))
        self._ca_x = self._pw_acc.plot(pen=pg.mkPen("#FF6B6B", width=1.5), name="Acc X")
        self._ca_y = self._pw_acc.plot(pen=pg.mkPen("#51CF66", width=1.5), name="Acc Y")
        self._ca_z = self._pw_acc.plot(pen=pg.mkPen("#74C0FC", width=1.5), name="Acc Z")
        ly.addWidget(self._pw_acc)

        # ── 角速度图 ──────────────────────────────────────────────────────
        self._pw_gyr = pg.PlotWidget(title="角速度 (rad/s)")
        self._pw_gyr.setLabel("left",   "rad/s")
        self._pw_gyr.setLabel("bottom", "采样点")
        self._pw_gyr.showGrid(x=False, y=True, alpha=0.25)
        self._pw_gyr.addLegend(offset=(5, 5))
        self._cg_x = self._pw_gyr.plot(pen=pg.mkPen("#FF6B6B", width=1.5), name="Gyr X")
        self._cg_y = self._pw_gyr.plot(pen=pg.mkPen("#51CF66", width=1.5), name="Gyr Y")
        self._cg_z = self._pw_gyr.plot(pen=pg.mkPen("#74C0FC", width=1.5), name="Gyr Z")
        ly.addWidget(self._pw_gyr)

    # ── 根据当前时间戳更新显示 ────────────────────────────────────────────────
    def update_to_time(self, imu_data: dict, t: float):
        times = imu_data["times"]
        end   = int(np.searchsorted(times, t, side="right"))
        start = max(0, end - self._hist)
        n     = end - start
        x     = np.arange(self._hist)

        if n <= 0:
            z = np.zeros(self._hist, np.float32)
            for c in (self._ca_x, self._ca_y, self._ca_z,
                      self._cg_x, self._cg_y, self._cg_z):
                c.setData(x, z)
            return

        def _slice(key):
            arr = imu_data[key][start:end]
            if n < self._hist:
                pad = self._hist - n
                arr = np.concatenate([np.full(pad, arr[0], np.float32), arr])
            return arr

        self._ca_x.setData(x, _slice("acc_x"))
        self._ca_y.setData(x, _slice("acc_y"))
        self._ca_z.setData(x, _slice("acc_z"))
        self._cg_x.setData(x, _slice("gyr_x"))
        self._cg_y.setData(x, _slice("gyr_y"))
        self._cg_z.setData(x, _slice("gyr_z"))

    def reset(self):
        x = np.arange(self._hist)
        z = np.zeros(self._hist, np.float32)
        for c in (self._ca_x, self._ca_y, self._ca_z,
                  self._cg_x, self._cg_y, self._cg_z):
            c.setData(x, z)

    def set_history(self, h: int):
        self._hist = h
        self.reset()


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────

class UIViewer(QWidget):
    def __init__(self, ult_file: str = "", imu_file: str = ""):
        super().__init__()

        # 数据状态
        self._data:          dict       = {}
        self._n_frames:      int        = 0
        self._cur_frame:     int        = 0
        self._master_times:  np.ndarray = np.array([])
        self._playing:       bool       = False

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        # 超声滚动缓冲
        self._ult_history: int  = 200
        self._ult_bufs:    dict = {}

        # 待加载路径
        self._pending_ult: str = ult_file
        self._pending_imu: str = imu_file

        # IMU 面板（按设备 id 动态创建）
        self._imu_panels: dict[str, IMUDevicePanel] = {}

        # pyqtgraph 图项
        self._img_items:    dict[int, pg.ImageItem]  = {}
        self._plot_widgets: dict[int, pg.PlotWidget] = {}

        self._load_thread = None
        self._build_ui()

        if ult_file or imu_file:
            self._do_load()

    # ── 界面构建 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("超声 + IMU 同步回放")
        self.resize(1300, 900)

        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(10, 10, 10, 10)

        # 标题
        lbl = QLabel("超声灰度图  +  IMU 动捕数据  同步回放")
        lbl.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(13); f.setBold(True)
        lbl.setFont(f)
        root.addWidget(lbl)

        # 文件加载行
        frow = QHBoxLayout()
        btn_ult = QPushButton("加载超声 CSV"); btn_ult.setFixedWidth(130)
        btn_imu = QPushButton("加载 IMU CSV");  btn_imu.setFixedWidth(130)
        btn_ult.clicked.connect(self._on_load_ult)
        btn_imu.clicked.connect(self._on_load_imu)
        self._lbl_ult = QLabel("超声：未加载"); self._lbl_ult.setStyleSheet("color:gray")
        self._lbl_imu = QLabel("IMU：未加载");  self._lbl_imu.setStyleSheet("color:gray")
        frow.addWidget(btn_ult); frow.addWidget(self._lbl_ult)
        frow.addSpacing(20)
        frow.addWidget(btn_imu); frow.addWidget(self._lbl_imu)
        frow.addStretch()
        root.addLayout(frow)

        # 状态行
        self._lbl_status = QLabel("请加载数据文件")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._lbl_status)

        # ── 主体区域：IMU（上）/ 超声（下），可拖拽分隔 ─────────────────────────
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # —— IMU 区域 ——
        pg.setConfigOptions(antialias=False, background="k", foreground="w")

        self._imu_group = QGroupBox("IMU 动捕（每设备一个面板  |  加速度 / 角速度）")
        self._imu_row   = QHBoxLayout(self._imu_group)
        self._imu_row.setSpacing(6)
        self._imu_placeholder = QLabel("暂无 IMU 数据，请加载 IMU CSV 文件")
        self._imu_placeholder.setAlignment(Qt.AlignCenter)
        self._imu_placeholder.setStyleSheet("color:gray")
        self._imu_row.addWidget(self._imu_placeholder)
        splitter.addWidget(self._imu_group)

        # —— 超声区域 ——
        ult_group = QGroupBox("超声灰度图（横轴=采样点，纵轴=帧序）")
        ult_ly = QVBoxLayout(ult_group); ult_ly.setSpacing(4)
        r1 = QHBoxLayout(); r1.setSpacing(4)
        r2 = QHBoxLayout(); r2.setSpacing(4)
        for ch in (1, 2, 3, 4):
            pw = pg.PlotWidget(title=f"通道 {ch}")
            pw.setLabel("bottom", "采样点")
            pw.setLabel("left",   "帧")
            img = pg.ImageItem()
            img.setLookupTable(self._gray_lut())
            pw.addItem(img)
            self._img_items[ch]    = img
            self._plot_widgets[ch] = pw
            (r1 if ch <= 2 else r2).addWidget(pw)
        ult_ly.addLayout(r1)
        ult_ly.addLayout(r2)
        splitter.addWidget(ult_group)
        splitter.setSizes([300, 480])

        # ── 播放控制 ──────────────────────────────────────────────────────────
        ctrl_group = QGroupBox("播放控制")
        ctrl_ly = QVBoxLayout(ctrl_group)

        # 进度滑条
        sl_row = QHBoxLayout()
        self._lbl_frame = QLabel("帧: 0 / 0"); self._lbl_frame.setFixedWidth(110)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0); self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider)
        self._lbl_time = QLabel("时间: 0.000 s"); self._lbl_time.setFixedWidth(130)
        sl_row.addWidget(self._lbl_frame)
        sl_row.addWidget(self._slider, stretch=1)
        sl_row.addWidget(self._lbl_time)
        ctrl_ly.addLayout(sl_row)

        # 按钮 + 参数行
        btn_row = QHBoxLayout()

        self._btn_play  = QPushButton("▶ 播放"); self._btn_play.setEnabled(False)
        self._btn_pause = QPushButton("⏸ 暂停"); self._btn_pause.setEnabled(False)
        self._btn_stop  = QPushButton("⏹ 停止")
        for b in (self._btn_play, self._btn_pause, self._btn_stop):
            b.setFixedHeight(32); b.setFixedWidth(90)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_stop.clicked.connect(self._on_stop)

        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(0.1, 500); self._spin_fps.setValue(10)
        self._spin_fps.setDecimals(1); self._spin_fps.setFixedWidth(75)
        self._spin_fps.setSuffix(" fps")
        self._spin_fps.valueChanged.connect(self._on_fps_changed)

        self._spin_ult_hist = QSpinBox()
        self._spin_ult_hist.setRange(10, 2000); self._spin_ult_hist.setValue(200)
        self._spin_ult_hist.setFixedWidth(70)
        self._spin_ult_hist.valueChanged.connect(self._on_ult_hist_changed)

        self._spin_imu_hist = QSpinBox()
        self._spin_imu_hist.setRange(10, 4000); self._spin_imu_hist.setValue(400)
        self._spin_imu_hist.setFixedWidth(70)
        self._spin_imu_hist.valueChanged.connect(self._on_imu_hist_changed)

        self._spin_smin = QSpinBox()
        self._spin_smin.setRange(0, 999);  self._spin_smin.setValue(0);    self._spin_smin.setFixedWidth(65)
        self._spin_smax = QSpinBox()
        self._spin_smax.setRange(1, 1000); self._spin_smax.setValue(1000); self._spin_smax.setFixedWidth(65)
        btn_s = QPushButton("应用"); btn_s.setFixedWidth(50); btn_s.setFixedHeight(28)
        btn_s.clicked.connect(self._apply_srange)

        self._spin_vmin = QDoubleSpinBox()
        self._spin_vmin.setRange(0, 1e6); self._spin_vmin.setValue(0)
        self._spin_vmin.setDecimals(0);   self._spin_vmin.setFixedWidth(75)
        self._spin_vmax = QDoubleSpinBox()
        self._spin_vmax.setRange(0, 1e6); self._spin_vmax.setValue(300)
        self._spin_vmax.setDecimals(0);   self._spin_vmax.setFixedWidth(75)
        btn_v = QPushButton("应用"); btn_v.setFixedWidth(50); btn_v.setFixedHeight(28)
        btn_v.clicked.connect(self._apply_vrange)

        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_pause)
        btn_row.addWidget(self._btn_stop)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("速度:")); btn_row.addWidget(self._spin_fps)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("超声帧数:")); btn_row.addWidget(self._spin_ult_hist)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("IMU窗口:")); btn_row.addWidget(self._spin_imu_hist)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("采样点:"))
        btn_row.addWidget(self._spin_smin); btn_row.addWidget(QLabel("~")); btn_row.addWidget(self._spin_smax)
        btn_row.addWidget(btn_s)
        btn_row.addSpacing(16)
        btn_row.addWidget(QLabel("灰度:"))
        btn_row.addWidget(self._spin_vmin); btn_row.addWidget(QLabel("~")); btn_row.addWidget(self._spin_vmax)
        btn_row.addWidget(btn_v)
        btn_row.addStretch()
        ctrl_ly.addLayout(btn_row)
        root.addWidget(ctrl_group)

    # ── 灰度 LUT ─────────────────────────────────────────────────────────────
    @staticmethod
    def _gray_lut():
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
        return lut

    # ── 文件加载 ──────────────────────────────────────────────────────────────
    def _on_load_ult(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择超声 CSV", str(Path.cwd()), "CSV (*.csv)")
        if p:
            self._pending_ult = p
            self._do_load()

    def _on_load_imu(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 IMU CSV", str(Path.cwd()), "CSV (*.csv)")
        if p:
            self._pending_imu = p
            self._do_load()

    def _do_load(self):
        self._on_stop()
        self._lbl_status.setText("正在加载...")
        self._load_thread = LoadThread(
            ult_path=self._pending_ult,
            imu_path=self._pending_imu,
        )
        self._load_thread.finished.connect(self._on_loaded)
        self._load_thread.start()

    def _on_loaded(self, result, err):
        if result is None:
            self._lbl_status.setText(f"加载失败: {err}"); return

        self._data = result

        # 更新标签
        if "ult" in result:
            chs = result["ult"]["channels"]
            n   = len(result["ult"]["times"][chs[0]])
            self._lbl_ult.setText(f"超声: 通道 {chs}  共 {n} 帧")
            self._lbl_ult.setStyleSheet("color:white")

        if "imu" in result:
            devs  = result["imu"]["devices"]
            n_smp = len(result["imu"]["data"][devs[0]]["times"])
            self._lbl_imu.setText(f"IMU: {len(devs)} 设备  各 {n_smp} 采样")
            self._lbl_imu.setStyleSheet("color:white")
            self._rebuild_imu_panels(devs)

        # 确定主时间轴（优先用超声；纯 IMU 回放时用 IMU 时间戳）
        if "ult" in result:
            chs = result["ult"]["channels"]
            self._master_times = result["ult"]["times"][chs[0]]
        elif "imu" in result:
            devs = result["imu"]["devices"]
            self._master_times = result["imu"]["data"][devs[0]]["times"]
        else:
            self._lbl_status.setText("数据为空"); return

        self._n_frames  = len(self._master_times)
        self._cur_frame = 0
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)

        self._init_ult_buffers()
        self._render_frame(0)

        t_span = (self._master_times[-1] - self._master_times[0]
                  if self._n_frames > 1 else 0.0)
        self._lbl_status.setText(
            f"已加载  {self._n_frames} 帧  时长 {t_span:.2f} s"
        )
        self._btn_play.setEnabled(True)

    # ── IMU 面板动态重建 ──────────────────────────────────────────────────────
    def _rebuild_imu_panels(self, devices: list):
        # 清除旧内容（包括占位符）
        while self._imu_row.count():
            item = self._imu_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._imu_panels.clear()

        h = self._spin_imu_hist.value()
        for dev in devices:
            panel = IMUDevicePanel(str(dev), history=h)
            self._imu_panels[str(dev)] = panel
            self._imu_row.addWidget(panel)

    # ── 超声滚动缓冲初始化 ────────────────────────────────────────────────────
    def _init_ult_buffers(self):
        self._ult_bufs = {}
        if "ult" not in self._data:
            return
        for ch in self._data["ult"]["channels"]:
            self._ult_bufs[ch] = np.zeros(
                (self._ult_history, 1000), dtype=np.float32
            )

    # ── 渲染帧 ────────────────────────────────────────────────────────────────
    def _render_frame(self, frame_idx: int):
        if not self._data or frame_idx >= self._n_frames:
            return

        cur_t = float(self._master_times[frame_idx])
        t0    = float(self._master_times[0])

        # ── 超声灰度图 ────────────────────────────────────────────────────
        if "ult" in self._data:
            vmin = float(self._spin_vmin.value())
            vmax = float(self._spin_vmax.value())
            if vmax <= vmin: vmax = vmin + 1
            s0 = max(0,    int(self._spin_smin.value()))
            s1 = min(1000, int(self._spin_smax.value()))
            if s1 <= s0: s1 = s0 + 1

            for ch in self._data["ult"]["channels"]:
                frm = self._data["ult"]["frames"][ch]
                buf = self._ult_bufs.get(ch)
                if buf is None:
                    continue
                if frame_idx < len(frm):
                    self._ult_bufs[ch] = np.roll(buf, -1, axis=0)
                    self._ult_bufs[ch][-1] = frm[frame_idx]
                    buf = self._ult_bufs[ch]

                sl  = buf[:, s0:s1]
                img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
                self._img_items[ch].setImage(img.T, autoLevels=False, levels=(0, 255))
                self._img_items[ch].setRect(
                    pg.QtCore.QRectF(s0, 0, s1 - s0, self._ult_history)
                )

        # ── IMU 折线图 ────────────────────────────────────────────────────
        if "imu" in self._data:
            for dev_id, panel in self._imu_panels.items():
                imu_d = self._data["imu"]["data"].get(dev_id)
                if imu_d:
                    panel.update_to_time(imu_d, cur_t)

        # ── 标签更新 ──────────────────────────────────────────────────────
        self._lbl_time.setText(f"时间: {cur_t - t0:.3f} s")
        self._lbl_frame.setText(f"帧: {frame_idx} / {self._n_frames - 1}")
        self._slider.setValue(frame_idx)
        self._cur_frame = frame_idx

    # ── 播放控制 ──────────────────────────────────────────────────────────────
    def _on_play(self):
        if not self._data: return
        if self._cur_frame >= self._n_frames - 1:
            self._cur_frame = 0
            self._init_ult_buffers()
        self._playing = True
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._play_timer.start(max(1, int(1000 / self._spin_fps.value())))

    def _on_pause(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)

    def _on_stop(self):
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setEnabled(bool(self._data))
        self._btn_pause.setEnabled(False)
        if self._data:
            self._cur_frame = 0
            self._init_ult_buffers()
            for p in self._imu_panels.values():
                p.reset()
            if self._n_frames > 0:
                self._render_frame(0)

    def _next_frame(self):
        if self._cur_frame >= self._n_frames - 1:
            self._on_stop(); return
        self._render_frame(self._cur_frame + 1)

    def _on_slider(self, val: int):
        self._on_pause()
        self._init_ult_buffers()
        start = max(0, val - self._ult_history + 1)
        for i in range(start, val + 1):
            self._render_frame(i)

    def _on_fps_changed(self, val: float):
        if self._playing:
            self._play_timer.setInterval(max(1, int(1000 / val)))

    def _on_ult_hist_changed(self, val: int):
        self._ult_history = val
        if self._data:
            self._init_ult_buffers()
            self._render_frame(self._cur_frame)

    def _on_imu_hist_changed(self, val: int):
        for panel in self._imu_panels.values():
            panel.set_history(val)
        if self._data:
            self._render_frame(self._cur_frame)

    def _apply_srange(self):
        if self._data:
            self._init_ult_buffers()
            self._render_frame(self._cur_frame)

    def _apply_vrange(self):
        if self._data:
            self._render_frame(self._cur_frame)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="超声 + IMU 同步回放")
    parser.add_argument("--ult", default="", help="超声 CSV 路径")
    parser.add_argument("--imu", default="", help="IMU CSV 路径")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = UIViewer(ult_file=args.ult, imu_file=args.imu)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
