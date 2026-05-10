"""
UI_viewer.py — 超声灰度图 + IMU + 电机数据同步回放
指定一个采集目录，自动识别其中的 ultrasound_*.csv / imu_*.csv / motor_*.csv，
用统一时间轴同步回放：
  · 超声：每通道一个灰度图（横轴=采样点，纵轴=帧）
  · IMU：每设备一个面板，上图=3轴加速度，下图=3轴角速度
  · 电机：左/右各一个面板，上图=速度，下图=编码器位置

用法：
    python UI_viewer.py
    python UI_viewer.py --dir ./data/S001_xxx
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
#  目录扫描：自动找各类 CSV
# ─────────────────────────────────────────────────────────────────────────────

def find_csv(directory: str, prefix: str) -> str:
    """在 directory 下找第一个以 prefix 开头的 .csv 文件，找不到返回空串。
    同时兼容 handle_data_XX/input/ 下的无后缀文件名（如 ultrasound.csv）。"""
    d = Path(directory)
    if not d.is_dir():
        return ""
    # 先找带时间戳后缀的（如 ultrasound_20260418.csv）
    matches = sorted(d.glob(f"{prefix}*.csv"))
    if matches:
        return str(matches[0])
    # 再找精确名称（去掉尾部下划线，如 ultrasound.csv）
    exact = d / f"{prefix.rstrip('_')}.csv"
    return str(exact) if exact.exists() else ""


# ─────────────────────────────────────────────────────────────────────────────
#  数据加载线程
# ─────────────────────────────────────────────────────────────────────────────

class LoadThread(QThread):
    finished = Signal(object, str)

    def __init__(self, ult_path="", imu_path="", motor_path=""):
        super().__init__()
        self._ult   = ult_path
        self._imu   = imu_path
        self._motor = motor_path

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
                        "times": s["timestamp"].values.astype(np.float64),
                        "acc_x": _col(s, "acc_x"), "acc_y": _col(s, "acc_y"), "acc_z": _col(s, "acc_z"),
                        "gyr_x": _col(s, "gyr_x"), "gyr_y": _col(s, "gyr_y"), "gyr_z": _col(s, "gyr_z"),
                    }
                res["imu"] = {"devices": [str(d) for d in devs], "data": dev_data}

            # ── 电机 ───────────────────────────────────────────────────────
            if self._motor:
                df = pd.read_csv(self._motor)

                def _mcol(name):
                    return df[name].values.astype(np.float32) if name in df.columns \
                           else np.zeros(len(df), np.float32)

                # 优先用硬件时间戳（ms→s），回退到 PC 时间戳
                if "timestamp_hw_ms" in df.columns:
                    hw = df["timestamp_hw_ms"].values.astype(np.float64)
                    t0 = hw[0]
                    times = (hw - t0) / 1000.0 + df["timestamp"].values.astype(np.float64)[0]
                else:
                    times = df["timestamp"].values.astype(np.float64)

                res["motor"] = {
                    "times": times,
                    "lpos":  _mcol("lpos"), "lvel":  _mcol("lvel"),
                    "rpos":  _mcol("rpos"), "rvel":  _mcol("rvel"),
                }

            self.finished.emit(res, "")
        except Exception as e:
            self.finished.emit(None, str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  通用滚动折线面板基类
# ─────────────────────────────────────────────────────────────────────────────

def _slice_window(arr, times, t, hist):
    """取 times <= t 的最后 hist 个点，不足时左填充首值。"""
    end   = int(np.searchsorted(times, t, side="right"))
    start = max(0, end - hist)
    n     = end - start
    x     = np.arange(hist)
    if n <= 0:
        return x, np.zeros(hist, np.float32)
    seg = arr[start:end]
    if n < hist:
        seg = np.concatenate([np.full(hist - n, seg[0], np.float32), seg])
    return x, seg


# ─────────────────────────────────────────────────────────────────────────────
#  IMU 设备面板
# ─────────────────────────────────────────────────────────────────────────────

class IMUDevicePanel(QGroupBox):
    def __init__(self, device_id: str, history: int = 400):
        super().__init__(f"IMU  —  {device_id}")
        self._hist = history
        ly = QVBoxLayout(self)
        ly.setContentsMargins(4, 14, 4, 4); ly.setSpacing(4)

        self._pw_acc = pg.PlotWidget(title="加速度 (m/s²)")
        self._pw_acc.setLabel("left", "m/s²", **{"font-size": "9px"})
        self._pw_acc.getAxis("left").setWidth(38)
        self._pw_acc.getAxis("bottom").setHeight(18)
        self._pw_acc.getPlotItem().titleLabel.setAttr("size", "9pt")
        self._pw_acc.showGrid(x=False, y=True, alpha=0.25)
        self._pw_acc.addLegend(offset=(2, 2), labelTextSize="8pt", colCount=3)
        self._ca_x = self._pw_acc.plot(pen=pg.mkPen("#FF6B6B", width=1.5), name="X")
        self._ca_y = self._pw_acc.plot(pen=pg.mkPen("#51CF66", width=1.5), name="Y")
        self._ca_z = self._pw_acc.plot(pen=pg.mkPen("#74C0FC", width=1.5), name="Z")
        ly.addWidget(self._pw_acc)

        self._pw_gyr = pg.PlotWidget(title="角速度 (rad/s)")
        self._pw_gyr.setLabel("left", "rad/s", **{"font-size": "9px"})
        self._pw_gyr.getAxis("left").setWidth(38)
        self._pw_gyr.getAxis("bottom").setHeight(18)
        self._pw_gyr.getPlotItem().titleLabel.setAttr("size", "9pt")
        self._pw_gyr.showGrid(x=False, y=True, alpha=0.25)
        self._pw_gyr.addLegend(offset=(2, 2), labelTextSize="8pt", colCount=3)
        self._cg_x = self._pw_gyr.plot(pen=pg.mkPen("#FF6B6B", width=1.5), name="X")
        self._cg_y = self._pw_gyr.plot(pen=pg.mkPen("#51CF66", width=1.5), name="Y")
        self._cg_z = self._pw_gyr.plot(pen=pg.mkPen("#74C0FC", width=1.5), name="Z")
        ly.addWidget(self._pw_gyr)

    def update_to_time(self, d: dict, t: float):
        for key, curve in [("acc_x", self._ca_x), ("acc_y", self._ca_y), ("acc_z", self._ca_z),
                           ("gyr_x", self._cg_x), ("gyr_y", self._cg_y), ("gyr_z", self._cg_z)]:
            x, y = _slice_window(d[key], d["times"], t, self._hist)
            curve.setData(x, y)

    def reset(self):
        x = np.arange(self._hist); z = np.zeros(self._hist, np.float32)
        for c in (self._ca_x, self._ca_y, self._ca_z, self._cg_x, self._cg_y, self._cg_z):
            c.setData(x, z)

    def set_history(self, h: int):
        self._hist = h; self.reset()


# ─────────────────────────────────────────────────────────────────────────────
#  电机面板（左 / 右各一个）
# ─────────────────────────────────────────────────────────────────────────────

class MotorPanel(QGroupBox):
    def __init__(self, side: str, history: int = 400):
        """side: 'left' or 'right'"""
        super().__init__(f"电机 — {'左' if side == 'left' else '右'}")
        self._hist  = history
        self._side  = side
        self._vkey  = "lvel" if side == "left" else "rvel"
        self._pkey  = "lpos" if side == "left" else "rpos"

        ly = QVBoxLayout(self)
        ly.setContentsMargins(4, 14, 4, 4); ly.setSpacing(4)

        self._pw_vel = pg.PlotWidget(title="速度 (rad/s)")
        self._pw_vel.setLabel("left", "rad/s", **{"font-size": "9px"})
        self._pw_vel.getAxis("left").setWidth(38)
        self._pw_vel.getAxis("bottom").setHeight(18)
        self._pw_vel.getPlotItem().titleLabel.setAttr("size", "9pt")
        self._pw_vel.showGrid(x=False, y=True, alpha=0.25)
        self._cv = self._pw_vel.plot(pen=pg.mkPen("#FFD43B", width=1.5))
        ly.addWidget(self._pw_vel)

        self._pw_pos = pg.PlotWidget(title="编码器位置 (rad)")
        self._pw_pos.setLabel("left", "rad", **{"font-size": "9px"})
        self._pw_pos.getAxis("left").setWidth(38)
        self._pw_pos.getAxis("bottom").setHeight(18)
        self._pw_pos.getPlotItem().titleLabel.setAttr("size", "9pt")
        self._pw_pos.showGrid(x=False, y=True, alpha=0.25)
        self._cp = self._pw_pos.plot(pen=pg.mkPen("#FF922B", width=1.5))
        ly.addWidget(self._pw_pos)

    def update_to_time(self, d: dict, t: float):
        x, vel = _slice_window(d[self._vkey], d["times"], t, self._hist)
        _,  pos = _slice_window(d[self._pkey], d["times"], t, self._hist)
        self._cv.setData(x, vel)
        self._cp.setData(x, pos)

    def reset(self):
        x = np.arange(self._hist); z = np.zeros(self._hist, np.float32)
        self._cv.setData(x, z); self._cp.setData(x, z)

    def set_history(self, h: int):
        self._hist = h; self.reset()


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────

class UIViewer(QWidget):
    def __init__(self, data_dir: str = ""):
        super().__init__()
        self._data:         dict       = {}
        self._n_frames:     int        = 0
        self._cur_frame:    int        = 0
        self._master_times: np.ndarray = np.array([])
        self._playing:      bool       = False

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._next_frame)

        self._ult_history: int  = 200
        self._ult_bufs:    dict = {}

        self._imu_panels:   dict[str, IMUDevicePanel] = {}
        self._motor_panels: dict[str, MotorPanel]     = {}

        self._img_items:    dict[int, pg.ImageItem]  = {}
        self._plot_widgets: dict[int, pg.PlotWidget] = {}

        self._load_thread = None
        self._data_dir    = data_dir

        self._build_ui()
        if data_dir:
            self._load_dir(data_dir)

    # ── 界面构建 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("超声 + IMU + 电机 同步回放")
        self.resize(1400, 960)

        root = QVBoxLayout(self)
        root.setSpacing(6); root.setContentsMargins(10, 10, 10, 10)

        lbl = QLabel("超声灰度图  +  IMU 动捕  +  电机数据  同步回放")
        lbl.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(13); f.setBold(True); lbl.setFont(f)
        root.addWidget(lbl)

        # 目录选择行
        drow = QHBoxLayout()
        btn_dir = QPushButton("选择采集目录"); btn_dir.setFixedWidth(130)
        btn_dir.clicked.connect(self._on_choose_dir)
        self._lbl_dir = QLabel("未选择目录"); self._lbl_dir.setStyleSheet("color:gray")
        drow.addWidget(btn_dir); drow.addWidget(self._lbl_dir, stretch=1)
        root.addLayout(drow)

        # 状态行
        self._lbl_status = QLabel("请选择采集目录")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._lbl_status)

        # ── 主体：三层可拖拽分隔（IMU / 电机 / 超声）────────────────────────
        pg.setConfigOptions(antialias=False, background="w", foreground="k")
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # IMU 区域
        self._imu_group = QGroupBox("IMU 动捕（每设备一个面板  |  加速度 / 角速度）")
        self._imu_row   = QHBoxLayout(self._imu_group); self._imu_row.setSpacing(6)
        ph = QLabel("暂无 IMU 数据"); ph.setAlignment(Qt.AlignCenter); ph.setStyleSheet("color:gray")
        self._imu_row.addWidget(ph)
        splitter.addWidget(self._imu_group)

        # 电机区域
        self._motor_group = QGroupBox("电机数据（左 / 右  |  速度 / 编码器位置）")
        self._motor_row   = QHBoxLayout(self._motor_group); self._motor_row.setSpacing(6)
        ph2 = QLabel("暂无电机数据"); ph2.setAlignment(Qt.AlignCenter); ph2.setStyleSheet("color:gray")
        self._motor_row.addWidget(ph2)
        splitter.addWidget(self._motor_group)

        # 超声区域
        ult_group = QGroupBox("超声灰度图（横轴=帧，纵轴=采样点，新帧从右侧进入）")
        ult_ly = QVBoxLayout(ult_group); ult_ly.setSpacing(4)
        r1 = QHBoxLayout(); r1.setSpacing(4)
        r2 = QHBoxLayout(); r2.setSpacing(4)
        for ch in (1, 2, 3, 4):
            pw = pg.PlotWidget()
            pw.setLabel("left", f"ch{ch} 采样点", **{"font-size": "9px"})
            pw.setLabel("bottom", "帧", **{"font-size": "9px"})
            pw.getAxis("left").setWidth(38)
            pw.getAxis("bottom").setHeight(18)
            pw.invertY(True)
            img = pg.ImageItem(); img.setLookupTable(self._gray_lut())
            pw.addItem(img)
            self._img_items[ch] = img; self._plot_widgets[ch] = pw
            (r1 if ch <= 2 else r2).addWidget(pw)
        ult_ly.addLayout(r1); ult_ly.addLayout(r2)
        splitter.addWidget(ult_group)
        splitter.setSizes([280, 280, 400])

        # ── 播放控制 ──────────────────────────────────────────────────────────
        ctrl_group = QGroupBox("播放控制")
        ctrl_ly = QVBoxLayout(ctrl_group)

        sl_row = QHBoxLayout()
        self._lbl_frame = QLabel("帧: 0 / 0"); self._lbl_frame.setFixedWidth(110)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0); self._slider.setMaximum(0)
        self._slider.sliderMoved.connect(self._on_slider)
        self._lbl_time = QLabel("时间: 0.000 s"); self._lbl_time.setFixedWidth(130)
        sl_row.addWidget(self._lbl_frame); sl_row.addWidget(self._slider, stretch=1); sl_row.addWidget(self._lbl_time)
        ctrl_ly.addLayout(sl_row)

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
        self._spin_fps.setDecimals(1); self._spin_fps.setFixedWidth(75); self._spin_fps.setSuffix(" fps")
        self._spin_fps.valueChanged.connect(self._on_fps_changed)

        self._spin_ult_hist = QSpinBox()
        self._spin_ult_hist.setRange(10, 2000); self._spin_ult_hist.setValue(200); self._spin_ult_hist.setFixedWidth(70)
        self._spin_ult_hist.valueChanged.connect(self._on_ult_hist_changed)

        self._spin_imu_hist = QSpinBox()
        self._spin_imu_hist.setRange(10, 4000); self._spin_imu_hist.setValue(400); self._spin_imu_hist.setFixedWidth(70)
        self._spin_imu_hist.valueChanged.connect(self._on_imu_hist_changed)

        self._spin_smin = QSpinBox(); self._spin_smin.setRange(0, 999);  self._spin_smin.setValue(0);    self._spin_smin.setFixedWidth(65)
        self._spin_smax = QSpinBox(); self._spin_smax.setRange(1, 1000); self._spin_smax.setValue(1000); self._spin_smax.setFixedWidth(65)
        btn_s = QPushButton("应用"); btn_s.setFixedWidth(50); btn_s.setFixedHeight(28); btn_s.clicked.connect(self._apply_srange)

        self._spin_vmin = QDoubleSpinBox(); self._spin_vmin.setRange(0, 1e6); self._spin_vmin.setValue(0);   self._spin_vmin.setDecimals(0); self._spin_vmin.setFixedWidth(75)
        self._spin_vmax = QDoubleSpinBox(); self._spin_vmax.setRange(0, 1e6); self._spin_vmax.setValue(300); self._spin_vmax.setDecimals(0); self._spin_vmax.setFixedWidth(75)
        btn_v = QPushButton("应用"); btn_v.setFixedWidth(50); btn_v.setFixedHeight(28); btn_v.clicked.connect(self._apply_vrange)

        btn_row.addWidget(self._btn_play); btn_row.addWidget(self._btn_pause); btn_row.addWidget(self._btn_stop)
        btn_row.addSpacing(16); btn_row.addWidget(QLabel("速度:")); btn_row.addWidget(self._spin_fps)
        btn_row.addSpacing(16); btn_row.addWidget(QLabel("超声帧数:")); btn_row.addWidget(self._spin_ult_hist)
        btn_row.addSpacing(16); btn_row.addWidget(QLabel("IMU/电机窗口:")); btn_row.addWidget(self._spin_imu_hist)
        btn_row.addSpacing(16); btn_row.addWidget(QLabel("采样点:"))
        btn_row.addWidget(self._spin_smin); btn_row.addWidget(QLabel("~")); btn_row.addWidget(self._spin_smax); btn_row.addWidget(btn_s)
        btn_row.addSpacing(16); btn_row.addWidget(QLabel("灰度:"))
        btn_row.addWidget(self._spin_vmin); btn_row.addWidget(QLabel("~")); btn_row.addWidget(self._spin_vmax); btn_row.addWidget(btn_v)
        btn_row.addStretch()
        ctrl_ly.addLayout(btn_row)
        root.addWidget(ctrl_group)

    # ── 灰度 LUT ─────────────────────────────────────────────────────────────
    @staticmethod
    def _gray_lut():
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[:, 0] = lut[:, 1] = lut[:, 2] = np.arange(256)
        return lut

    # ── 目录选择 ──────────────────────────────────────────────────────────────
    def _on_choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择采集目录", str(Path.cwd()))
        if d:
            self._load_dir(d)

    def _load_dir(self, directory: str):
        self._on_stop()
        ult_path   = find_csv(directory, "ultrasound_")
        imu_path   = find_csv(directory, "imu_")
        motor_path = find_csv(directory, "motor_")

        found = [n for n, p in [("超声", ult_path), ("IMU", imu_path), ("电机", motor_path)] if p]
        self._lbl_dir.setText(Path(directory).name)
        self._lbl_dir.setStyleSheet("color:white")
        self._lbl_status.setText(f"正在加载：{', '.join(found) if found else '未找到任何数据文件'}...")

        self._load_thread = LoadThread(ult_path, imu_path, motor_path)
        self._load_thread.finished.connect(self._on_loaded)
        self._load_thread.start()

    def _on_loaded(self, result, err):
        if result is None:
            self._lbl_status.setText(f"加载失败: {err}"); return

        self._data = result
        parts = []

        if "ult" in result:
            chs = result["ult"]["channels"]
            parts.append(f"超声 {len(result['ult']['times'][chs[0]])} 帧")

        if "imu" in result:
            devs = result["imu"]["devices"]
            parts.append(f"IMU {len(devs)} 设备 {len(result['imu']['data'][devs[0]]['times'])} 采样")
            self._rebuild_imu_panels(devs)

        if "motor" in result:
            parts.append(f"电机 {len(result['motor']['times'])} 帧")
            self._rebuild_motor_panels()

        if "ult" in result:
            chs = result["ult"]["channels"]
            self._master_times = result["ult"]["times"][chs[0]]
        elif "imu" in result:
            devs = result["imu"]["devices"]
            self._master_times = result["imu"]["data"][devs[0]]["times"]
        elif "motor" in result:
            self._master_times = result["motor"]["times"]
        else:
            self._lbl_status.setText("目录中未找到任何数据文件"); return

        self._n_frames  = len(self._master_times)
        self._cur_frame = 0
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self._slider.setValue(0)
        self._init_ult_buffers()
        self._render_frame(0)

        t_span = (self._master_times[-1] - self._master_times[0]) if self._n_frames > 1 else 0.0
        self._lbl_status.setText(f"已加载  {' | '.join(parts)}  时长 {t_span:.2f} s")
        self._btn_play.setEnabled(True)

    # ── 面板重建 ──────────────────────────────────────────────────────────────
    def _rebuild_imu_panels(self, devices: list):
        while self._imu_row.count():
            item = self._imu_row.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._imu_panels.clear()
        h = self._spin_imu_hist.value()
        for dev in devices:
            panel = IMUDevicePanel(str(dev), history=h)
            self._imu_panels[str(dev)] = panel
            self._imu_row.addWidget(panel)

    def _rebuild_motor_panels(self):
        while self._motor_row.count():
            item = self._motor_row.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._motor_panels.clear()
        h = self._spin_imu_hist.value()
        for side in ("left", "right"):
            panel = MotorPanel(side, history=h)
            self._motor_panels[side] = panel
            self._motor_row.addWidget(panel)

    # ── 超声缓冲 ──────────────────────────────────────────────────────────────
    def _init_ult_buffers(self):
        self._ult_bufs = {}
        if "ult" not in self._data: return
        for ch in self._data["ult"]["channels"]:
            self._ult_bufs[ch] = np.zeros((self._ult_history, 1000), dtype=np.float32)

    # ── 渲染帧 ────────────────────────────────────────────────────────────────
    def _render_frame(self, frame_idx: int):
        if not self._data or frame_idx >= self._n_frames: return
        cur_t = float(self._master_times[frame_idx])
        t0    = float(self._master_times[0])

        if "ult" in self._data:
            vmin = float(self._spin_vmin.value()); vmax = float(self._spin_vmax.value())
            if vmax <= vmin: vmax = vmin + 1
            s0 = max(0, int(self._spin_smin.value())); s1 = min(1000, int(self._spin_smax.value()))
            if s1 <= s0: s1 = s0 + 1
            for ch in self._data["ult"]["channels"]:
                frm = self._data["ult"]["frames"][ch]
                buf = self._ult_bufs.get(ch)
                if buf is None: continue
                if frame_idx < len(frm):
                    self._ult_bufs[ch] = np.roll(buf, -1, axis=0)
                    self._ult_bufs[ch][-1] = frm[frame_idx]
                    buf = self._ult_bufs[ch]
                sl  = buf[:, s0:s1]
                img = (np.clip((sl - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
                # img shape: (history, samples) → setImage expects (x, y) = (frames, samples)
                self._img_items[ch].setImage(img, autoLevels=False, levels=(0, 255))
                self._img_items[ch].setRect(pg.QtCore.QRectF(0, s0, self._ult_history, s1 - s0))

        if "imu" in self._data:
            for dev_id, panel in self._imu_panels.items():
                d = self._data["imu"]["data"].get(dev_id)
                if d: panel.update_to_time(d, cur_t)

        if "motor" in self._data:
            for panel in self._motor_panels.values():
                panel.update_to_time(self._data["motor"], cur_t)

        self._lbl_time.setText(f"时间: {cur_t - t0:.3f} s")
        self._lbl_frame.setText(f"帧: {frame_idx} / {self._n_frames - 1}")
        self._slider.setValue(frame_idx)
        self._cur_frame = frame_idx

    # ── 播放控制 ──────────────────────────────────────────────────────────────
    def _on_play(self):
        if not self._data: return
        if self._cur_frame >= self._n_frames - 1:
            self._cur_frame = 0; self._init_ult_buffers()
        self._playing = True
        self._btn_play.setEnabled(False); self._btn_pause.setEnabled(True)
        self._play_timer.start(max(1, int(1000 / self._spin_fps.value())))

    def _on_pause(self):
        self._playing = False; self._play_timer.stop()
        self._btn_play.setEnabled(True); self._btn_pause.setEnabled(False)

    def _on_stop(self):
        self._playing = False; self._play_timer.stop()
        self._btn_play.setEnabled(bool(self._data)); self._btn_pause.setEnabled(False)
        if self._data:
            self._cur_frame = 0; self._init_ult_buffers()
            for p in self._imu_panels.values():   p.reset()
            for p in self._motor_panels.values(): p.reset()
            if self._n_frames > 0: self._render_frame(0)

    def _next_frame(self):
        if self._cur_frame >= self._n_frames - 1:
            self._on_stop(); return
        self._render_frame(self._cur_frame + 1)

    def _on_slider(self, val: int):
        self._on_pause(); self._init_ult_buffers()
        for i in range(max(0, val - self._ult_history + 1), val + 1):
            self._render_frame(i)

    def _on_fps_changed(self, val: float):
        if self._playing: self._play_timer.setInterval(max(1, int(1000 / val)))

    def _on_ult_hist_changed(self, val: int):
        self._ult_history = val
        if self._data:
            self._init_ult_buffers()
            self._render_frame(self._cur_frame)

    def _on_imu_hist_changed(self, val: int):
        for p in self._imu_panels.values():   p.set_history(val)
        for p in self._motor_panels.values(): p.set_history(val)
        if self._data: self._render_frame(self._cur_frame)

    def _apply_srange(self):
        if self._data:
            self._init_ult_buffers()
            self._render_frame(self._cur_frame)

    def _apply_vrange(self):
        if self._data: self._render_frame(self._cur_frame)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="超声 + IMU + 电机 同步回放")
    parser.add_argument("--dir", default="", help="采集目录路径")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = UIViewer(data_dir=args.dir)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
