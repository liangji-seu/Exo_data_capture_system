#!/usr/bin/env python3
"""
gait_labeler.py — 步态相位 & 力矩手动标注工具（含 IMU 自动检测）

使用流程：
  1. 打开超声 CSV（ultrasound_xxx.csv），同目录 imu.csv 自动识别
  2. 点击"自动检测步态边界"从 IMU gyr_z 自动标定周期边界（黄线）
  3. 手动调整：左键 B-mode 添加边界，右键删除最近边界
  4. 力矩由 LUT 样条（hip_torque_lut.csv）从相位自动计算，无需手动绘制
  5. 保存：每个通道单独存一个 CSV
     格式：d0..d299（原始 RF，ROI 400~700）+ torque + phase

力矩算法与 gait_verify_manager / gait_torque_simulator.py 完全一致：
  phase(%) → LUT 线性插值 × 体重系数 → 低通滤波
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace, median_filter
from scipy.signal import hilbert

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QMessageBox,
        QDoubleSpinBox, QGroupBox, QCheckBox,
    )
    _Qt = Qt
except ImportError:
    from PyQt5.QtCore import Qt, QThread
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QMessageBox,
        QDoubleSpinBox, QGroupBox, QCheckBox,
    )
    _Qt = Qt

import pyqtgraph as pg

pg.setConfigOptions(antialias=False, background="k", foreground="w")

# ── 常量 ─────────────────────────────────────────────────────────────────────
ROI_START = 400
ROI_END   = 700
ROI_N     = ROI_END - ROI_START   # 300

LEFT_IMU_SUFFIX          = "260E"
DEFAULT_STRIDE_PERIOD    = 1.2
SWING_THRESHOLD          = 0.5
MOTION_TIMEOUT           = 2.0
STRIDE_PERIOD_MIN        = 0.5
STRIDE_PERIOD_MAX        = 2.5
STRIDE_PERIOD_FILTER_OLD = 0.8
STRIDE_PERIOD_FILTER_NEW = 0.2
SWING_EXIT_FACTOR        = 0.5
MOTION_DETECT_FACTOR     = 0.3
PHASE_RESTART_THRESHOLD  = 0.8
REFERENCE_WEIGHT         = 72.0
DEFAULT_PHASE_OFFSET     = 60.0
DEFAULT_FILTER_ALPHA     = 0.3
DT_CLAMP_MAX             = 0.1
DT_CLAMP_DEFAULT         = 0.01

LUT_PATH      = Path(__file__).parent.parent / "gait_verify_manager" / "src" / "hip_torque_lut.csv"
DATA_SAVE_DIR = Path(__file__).parent / "data"   # second_train/data/

# 默认打开目录：自动找 超声采集数据/第二批*/
_CAPTURE_ROOT = Path(__file__).parent.parent / "超声采集数据"
_DEFAULT_OPEN_DIR: Path = _CAPTURE_ROOT
if _CAPTURE_ROOT.exists():
    _second = next(
        (d for d in sorted(_CAPTURE_ROOT.iterdir()) if d.is_dir() and "第二批" in d.name),
        _CAPTURE_ROOT,
    )
    _DEFAULT_OPEN_DIR = _second


# ─────────────────────────────────────────────────────────────────────────────
#  步态检测器（直接移植自 gait_torque_simulator.py / KneeGaitDetector）
# ─────────────────────────────────────────────────────────────────────────────

class KneeGaitDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_velocity_    = 0.0
        self.phase_            = 0.0
        self.last_peak_time_   = 0.0
        self.time_             = 0.0
        self.stride_period_    = DEFAULT_STRIDE_PERIOD
        self.in_swing_         = False
        self.last_motion_time_ = 0.0

    def update(self, angular_velocity_z: float, dt: float) -> float:
        self.time_ += dt

        if abs(angular_velocity_z) > SWING_THRESHOLD * MOTION_DETECT_FACTOR:
            self.last_motion_time_ = self.time_

        if self.time_ - self.last_motion_time_ > MOTION_TIMEOUT:
            self.phase_ = 0.0
            self.prev_velocity_ = angular_velocity_z
            return self.phase_

        if not self.in_swing_ and abs(angular_velocity_z) > SWING_THRESHOLD:
            self.in_swing_ = True
            if self.last_peak_time_ > 0.0 and self.phase_ > PHASE_RESTART_THRESHOLD:
                period = self.time_ - self.last_peak_time_
                if STRIDE_PERIOD_MIN < period < STRIDE_PERIOD_MAX:
                    self.stride_period_ = (STRIDE_PERIOD_FILTER_OLD * self.stride_period_
                                          + STRIDE_PERIOD_FILTER_NEW * period)
                self.last_peak_time_ = self.time_
                self.phase_ = 0.0
            elif self.last_peak_time_ == 0.0:
                self.last_peak_time_ = self.time_
                self.phase_ = 0.0
        elif self.in_swing_ and abs(angular_velocity_z) < SWING_THRESHOLD * SWING_EXIT_FACTOR:
            self.in_swing_ = False

        if self.last_peak_time_ > 0.0:
            elapsed = self.time_ - self.last_peak_time_
            lp = min(elapsed / self.stride_period_, 1.0)
            self.phase_ = lp * lp * lp * (lp * (lp * 6.0 - 15.0) + 10.0)

        self.prev_velocity_ = angular_velocity_z
        return self.phase_


# ─────────────────────────────────────────────────────────────────────────────
#  髋关节力矩查找表（移植自 gait_torque_simulator.py / HipTorqueLUT）
# ─────────────────────────────────────────────────────────────────────────────

class HipTorqueLUT:
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path, header=None)
        self._data = df.iloc[:, 1].values.astype(float)
        assert len(self._data) == 1001, f"LUT 应有 1001 行，实际 {len(self._data)}"

    def get_torque(self, phase_percent: float) -> float:
        phase_percent = max(0.0, min(100.0, phase_percent))
        idx = phase_percent * 10.0
        i0  = int(idx)
        i1  = min(i0 + 1, len(self._data) - 1)
        t   = idx - i0
        return self._data[i0] + t * (self._data[i1] - self._data[i0])

    def batch(self, phase_arr: np.ndarray) -> np.ndarray:
        return np.array([self.get_torque(float(p)) for p in phase_arr])


# ─────────────────────────────────────────────────────────────────────────────
#  超声预处理
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_channel(raw: np.ndarray,
                         tgc_slope=0.025, sigma_depth=1.2,
                         sigma_time=0.8, sharpen=0.2) -> np.ndarray:
    raw_f = raw.astype(np.float32)
    n_frames, n_samples = raw_f.shape
    rf       = raw_f - raw_f.mean(axis=1, keepdims=True)
    envelope = np.abs(hilbert(rf, axis=1))
    log_env  = 20.0 * np.log10(np.clip(envelope, 1e-6, None))
    out      = np.zeros_like(log_env)
    tgc_gain = np.arange(n_samples) * tgc_slope
    for i in range(n_frames):
        frame = log_env[i] + tgc_gain
        vmin  = np.percentile(frame, 25.0)
        vmax  = np.percentile(frame, 99.0)
        out[i] = np.clip((frame - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0)
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


class _ProcessThread(QThread):
    finished = Signal(dict)

    def __init__(self, raw_frames: dict):
        super().__init__()
        self._raw = raw_frames

    def run(self):
        result = {}
        for ch, frames in self._raw.items():
            result[ch] = _preprocess_channel(frames)
        self.finished.emit(result)


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────

class GaitLabeler(QWidget):
    def __init__(self, csv_file: str = ""):
        super().__init__()
        self.setWindowTitle("Gait Phase & Torque Labeler — 步态标注工具")
        self.resize(1500, 980)

        # 数据
        self._raw_frames:    dict = {}
        self._proc_frames:   dict = {}
        self._us_timestamps: np.ndarray = np.array([])
        self._n_frames:      int = 0
        self._csv_path:      str = ""
        self._session_dir:   Path | None = None   # 采集目录（s_XXXXXXXX）
        self._imu_path:      Path | None = None   # 同目录 imu_*.csv

        # 相位边界（帧索引列表，已排序）
        self._phase_markers: list = []

        # 计算结果
        self._phase_arr:  np.ndarray = np.array([])
        self._torque_arr: np.ndarray = np.array([])

        # LUT
        self._lut: HipTorqueLUT | None = None
        if LUT_PATH.exists():
            try:
                self._lut = HipTorqueLUT(LUT_PATH)
            except Exception as e:
                print(f"[WARN] LUT 加载失败: {e}")

        self._build_ui()

        if csv_file:
            self._load_file(csv_file)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # 顶栏第一行：文件操作
        row1 = QHBoxLayout()
        btn_open = QPushButton("📂 打开采集目录")
        btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("未加载 — 请选择 s_XXXXXXXX 采集目录")
        self._lbl_file.setStyleSheet("color:#888")
        row1.addWidget(btn_open)
        row1.addWidget(self._lbl_file, stretch=1)

        self._btn_auto = QPushButton("🤖 自动检测步态边界（IMU）")
        self._btn_auto.setEnabled(False)
        self._btn_auto.clicked.connect(self._on_auto_detect)
        row1.addWidget(self._btn_auto)

        btn_clear = QPushButton("清除所有边界")
        btn_clear.clicked.connect(self._clear_phase_markers)
        row1.addWidget(btn_clear)

        # 通道勾选框
        row1.addWidget(QLabel("保存通道:"))
        self._ch_checks: dict[int, QCheckBox] = {}
        for ch in range(1, 5):
            cb = QCheckBox(str(ch))
            cb.setChecked(True)
            self._ch_checks[ch] = cb
            row1.addWidget(cb)

        btn_save = QPushButton("💾 保存数据集 CSV")
        btn_save.setStyleSheet("font-weight:bold; color:#00ff00; background:#333; padding:4px 10px")
        btn_save.clicked.connect(self._on_save)
        row1.addWidget(btn_save)
        root.addLayout(row1)

        # 顶栏第二行：力矩参数
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("力矩参数:"))

        grp = QGroupBox()
        grp.setMaximumHeight(54)
        grp_ly = QHBoxLayout(grp)
        grp_ly.setContentsMargins(6, 2, 6, 2)
        grp_ly.setSpacing(8)

        grp_ly.addWidget(QLabel("体重(kg):"))
        self._spin_weight = QDoubleSpinBox()
        self._spin_weight.setRange(30, 150); self._spin_weight.setValue(72); self._spin_weight.setFixedWidth(65)
        self._spin_weight.valueChanged.connect(self._recompute_torque)
        grp_ly.addWidget(self._spin_weight)

        grp_ly.addWidget(QLabel("相位偏移(%):"))
        self._spin_offset = QDoubleSpinBox()
        self._spin_offset.setRange(0, 100); self._spin_offset.setValue(DEFAULT_PHASE_OFFSET); self._spin_offset.setFixedWidth(65)
        self._spin_offset.valueChanged.connect(self._recompute_torque)
        grp_ly.addWidget(self._spin_offset)

        grp_ly.addWidget(QLabel("滤波α:"))
        self._spin_alpha = QDoubleSpinBox()
        self._spin_alpha.setRange(0.01, 1.0); self._spin_alpha.setValue(DEFAULT_FILTER_ALPHA)
        self._spin_alpha.setDecimals(2); self._spin_alpha.setFixedWidth(60)
        self._spin_alpha.valueChanged.connect(self._recompute_torque)
        grp_ly.addWidget(self._spin_alpha)

        row2.addWidget(grp)

        help_lbl = QLabel(
            "  左键 B-mode 图 → 添加边界（黄线）  |  右键 → 删除最近边界  |  "
            "调整参数后力矩自动更新"
        )
        help_lbl.setStyleSheet("color:#666; font-size:11px")
        row2.addWidget(help_lbl, stretch=1)
        root.addLayout(row2)

        # 主区域
        main = QHBoxLayout()
        main.setSpacing(6)

        # 左：4 通道 B-mode
        left_ly = QVBoxLayout()
        left_ly.setSpacing(3)
        lut = _gray_lut()
        self._bmode_plots:  dict = {}
        self._bmode_imgs:   dict = {}
        self._bmode_vlines: dict = {}

        for ch in range(1, 5):
            pw = pg.PlotWidget(title=f"Channel {ch}   (全深度 0~1000)")
            pw.setLabel("bottom", "帧索引")
            pw.setLabel("left",   "深度（采样点）")
            pw.invertY(True)
            pw.setMinimumHeight(155)
            pw.setMaximumHeight(195)

            img = pg.ImageItem()
            img.setLookupTable(lut)
            pw.addItem(img)

            self._bmode_plots[ch] = pw
            self._bmode_imgs[ch]  = img
            self._bmode_vlines[ch] = []

            def _make_handler(channel, plot_widget):
                def handler(evt):
                    btn = evt.button()
                    pos = plot_widget.plotItem.vb.mapSceneToView(evt.scenePos())
                    fidx = int(round(pos.x()))
                    if btn == _Qt.LeftButton:
                        self._add_phase_marker(fidx)
                    elif btn == _Qt.RightButton:
                        self._remove_nearest_phase_marker(fidx)
                return handler

            pw.scene().sigMouseClicked.connect(_make_handler(ch, pw))
            left_ly.addWidget(pw)

        main.addLayout(left_ly, stretch=3)

        # 右：相位图 + 力矩图
        right_ly = QVBoxLayout()
        right_ly.setSpacing(4)

        self._phase_plot = pg.PlotWidget(title="步态相位 (%)")
        self._phase_plot.setLabel("bottom", "帧索引")
        self._phase_plot.setLabel("left",   "Phase (%)")
        self._phase_plot.setYRange(0, 105)
        self._phase_curve   = self._phase_plot.plot(pen=pg.mkPen("#FF5722", width=2))
        self._phase_pvlines: list = []
        right_ly.addWidget(self._phase_plot)

        self._torque_plot = pg.PlotWidget(title="髋关节力矩（LUT 样条，调参数自动更新）")
        self._torque_plot.setLabel("bottom", "帧索引")
        self._torque_plot.setLabel("left",   "Torque (Nm/kg × 0.001)")
        self._torque_curve = self._torque_plot.plot(pen=pg.mkPen("#4CAF50", width=2))
        right_ly.addWidget(self._torque_plot)

        main.addLayout(right_ly, stretch=2)
        root.addLayout(main, stretch=1)

        # 状态栏
        self._lbl_status = QLabel("就绪")
        self._lbl_status.setStyleSheet("color:#aaa; font-size:11px")
        root.addWidget(self._lbl_status)

        # 链接 x 轴
        for ch in [2, 3, 4]:
            self._bmode_plots[ch].setXLink(self._bmode_plots[1])
        self._phase_plot.setXLink(self._bmode_plots[1])
        self._torque_plot.setXLink(self._bmode_plots[1])

    # ── 文件加载 ──────────────────────────────────────────────────────────────

    def _on_open(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "选择采集目录（s_XXXXXXXX）", str(_DEFAULT_OPEN_DIR)
        )
        if chosen:
            self._load_session_dir(Path(chosen))

    def _load_session_dir(self, session_dir: Path):
        # 找超声 CSV：优先 ultrasound.csv，兜底 ultrasound_*.csv
        us_path = (session_dir / "ultrasound.csv")
        if not us_path.exists():
            candidates = sorted(session_dir.glob("ultrasound_*.csv"))
            if not candidates:
                QMessageBox.critical(self, "错误", f"目录内未找到 ultrasound.csv:\n{session_dir}")
                return
            us_path = candidates[0]

        # 找 IMU CSV：优先 imu.csv，兜底 imu_*.csv
        imu_path = session_dir / "imu.csv"
        if not imu_path.exists():
            imu_candidates = sorted(session_dir.glob("imu_*.csv"))
            imu_path = imu_candidates[0] if imu_candidates else None

        self._imu_path    = imu_path
        self._session_dir = session_dir
        self._load_file(str(us_path))

    def _load_file(self, path: str):
        self._lbl_file.setText("加载中…")
        self._lbl_status.setText("读取 CSV…")
        QApplication.processEvents()

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取:\n{e}")
            return

        dcols = sorted(
            [c for c in df.columns if c.startswith("d") and c[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        if not dcols:
            QMessageBox.critical(self, "错误", "CSV 中未找到 d0..dN 格式列")
            return

        self._raw_frames    = {}
        self._us_timestamps = np.array([])
        self._csv_path      = path

        for ch in sorted(df["channel"].unique()):
            sub = df[df["channel"] == ch].reset_index(drop=True)
            self._raw_frames[int(ch)] = sub[dcols].values.astype(np.float32)
            if self._us_timestamps.size == 0 and "timestamp" in sub.columns:
                self._us_timestamps = sub["timestamp"].values.astype(np.float64)

        self._n_frames = len(next(iter(self._raw_frames.values())))
        self._phase_markers.clear()
        self._phase_arr  = np.zeros(self._n_frames, dtype=np.float32)
        self._torque_arr = np.zeros(self._n_frames, dtype=np.float32)

        self._lbl_file.setText(
            f"{self._session_dir.name} / {Path(path).name}   ({self._n_frames} 帧，{len(self._raw_frames)} 通道)"
            if self._session_dir else
            f"{Path(path).name}   ({self._n_frames} 帧，{len(self._raw_frames)} 通道)"
        )
        self._btn_auto.setEnabled(True)
        self._lbl_status.setText("预处理中，请稍候…")

        self._thread = _ProcessThread(self._raw_frames)
        self._thread.finished.connect(self._on_proc_done)
        self._thread.start()

    def _on_proc_done(self, result: dict):
        self._proc_frames = result
        self._update_bmode()
        self._update_phase_display()
        self._recompute_torque()

        if self._imu_path:
            self._lbl_status.setText(
                f"加载完成 — 发现 {self._imu_path.name}，可点击「自动检测」  |  "
                "左键添加边界，右键删除"
            )
        else:
            self._lbl_status.setText(
                "加载完成 — 未找到 imu_*.csv，请手动标注步态边界"
            )

    # ── B-mode 显示 ───────────────────────────────────────────────────────────

    def _update_bmode(self):
        if not self._proc_frames:
            return
        n_depth = next(iter(self._proc_frames.values())).shape[1]   # 1000
        for ch, frames in self._proc_frames.items():
            img_data = (frames * 255).astype(np.uint8)              # (N, 1000)
            self._bmode_imgs[ch].setImage(img_data, autoLevels=False, levels=(0, 255))
            self._bmode_plots[ch].setXRange(0, self._n_frames - 1, padding=0.01)
            self._bmode_plots[ch].setYRange(0, n_depth - 1,        padding=0.01)
        self._redraw_all_vlines()

    # ── 自动检测（IMU） ───────────────────────────────────────────────────────

    def _on_auto_detect(self):
        # 直接用加载时已找到的 imu 路径，找不到再弹对话框
        if self._imu_path and self._imu_path.exists():
            imu_path = str(self._imu_path)
        else:
            imu_path, _ = QFileDialog.getOpenFileName(
                self, "选择 IMU CSV",
                str(self._session_dir or Path(self._csv_path).parent),
                "CSV (*.csv)",
            )
        if not imu_path:
            return

        self._lbl_status.setText("正在从 IMU 检测步态边界…")
        QApplication.processEvents()

        try:
            imu_df = pd.read_csv(imu_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取 IMU:\n{e}")
            return

        mask = imu_df["device_id"].astype(str).str.upper().str.contains(LEFT_IMU_SUFFIX.upper())
        left_df = imu_df[mask].reset_index(drop=True)
        if left_df.empty:
            QMessageBox.critical(self, "错误", f"未找到尾号含 {LEFT_IMU_SUFFIX} 的 IMU 数据")
            return

        gyr_z = left_df["gyr_z"].values.astype(float)
        imu_ts = left_df["timestamp"].values.astype(float)

        # 运行检测器，记录每次相位复位的时间戳
        detector = KneeGaitDetector()
        boundary_ts: list[float] = []
        prev_phase = 0.0

        for i in range(len(gyr_z)):
            dt = DT_CLAMP_DEFAULT if i == 0 else float(imu_ts[i] - imu_ts[i - 1])
            if dt <= 0 or dt > DT_CLAMP_MAX:
                dt = DT_CLAMP_DEFAULT
            phase = detector.update(-gyr_z[i], dt)

            # 检测相位从高（>PHASE_RESTART_THRESHOLD）复位到近 0
            if prev_phase > PHASE_RESTART_THRESHOLD and phase < 0.05:
                boundary_ts.append(float(imu_ts[i]))
            prev_phase = phase

        if not boundary_ts:
            QMessageBox.warning(self, "未检测到边界", "未检测到步态周期边界，请手动标注")
            return

        # 将时间戳映射到最近的超声帧索引
        if self._us_timestamps.size == 0:
            QMessageBox.warning(self, "无时间戳", "超声 CSV 中无 timestamp 列，无法自动对齐")
            return

        new_markers = []
        for bts in boundary_ts:
            fidx = int(np.argmin(np.abs(self._us_timestamps - bts)))
            if fidx not in new_markers:
                new_markers.append(fidx)

        self._phase_markers = sorted(new_markers)
        self._update_phase_display()
        self._redraw_all_vlines()
        self._recompute_torque()

        self._lbl_status.setText(
            f"自动检测完成 — 共 {len(self._phase_markers)} 个边界  |  "
            "右键点击边界线可删除，左键可添加新边界"
        )

    # ── 相位边界管理 ─────────────────────────────────────────────────────────

    def _add_phase_marker(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self._n_frames:
            return
        if frame_idx not in self._phase_markers:
            self._phase_markers.append(frame_idx)
            self._phase_markers.sort()
        self._update_phase_display()
        self._redraw_all_vlines()
        self._recompute_torque()
        self._lbl_status.setText(
            f"添加边界 @ 帧 {frame_idx}  |  共 {len(self._phase_markers)} 个"
        )

    def _remove_nearest_phase_marker(self, frame_idx: int):
        if not self._phase_markers:
            return
        tol     = max(15, self._n_frames // 80)
        nearest = min(self._phase_markers, key=lambda m: abs(m - frame_idx))
        if abs(nearest - frame_idx) <= tol:
            self._phase_markers.remove(nearest)
            self._update_phase_display()
            self._redraw_all_vlines()
            self._recompute_torque()
            self._lbl_status.setText(
                f"删除边界 @ 帧 {nearest}  |  剩余 {len(self._phase_markers)} 个"
            )

    def _clear_phase_markers(self):
        self._phase_markers.clear()
        self._phase_arr  = np.zeros(self._n_frames, dtype=np.float32)
        self._torque_arr = np.zeros(self._n_frames, dtype=np.float32)
        self._update_phase_display()
        self._redraw_all_vlines()
        self._update_torque_display()
        self._lbl_status.setText("已清除所有边界标注")

    # ── 相位计算 & 显示 ───────────────────────────────────────────────────────

    def _compute_phase(self) -> np.ndarray:
        phase   = np.zeros(self._n_frames, dtype=np.float32)
        markers = sorted(self._phase_markers)
        if len(markers) >= 2:
            for i in range(len(markers) - 1):
                s, e = markers[i], markers[i + 1]
                if e > s:
                    phase[s:e] = np.linspace(0.0, 100.0, e - s, endpoint=False)
            # 最后边界之后延续 100%
            if markers[-1] < self._n_frames - 1:
                phase[markers[-1]:] = 100.0
        return phase

    def _update_phase_display(self):
        if self._n_frames == 0:
            return
        self._phase_arr = self._compute_phase()
        x = np.arange(self._n_frames)
        self._phase_curve.setData(x, self._phase_arr)

        for line in self._phase_pvlines:
            self._phase_plot.removeItem(line)
        self._phase_pvlines.clear()
        for m in self._phase_markers:
            line = self._phase_plot.addLine(
                x=m,
                pen=pg.mkPen("#FFFF00", width=1.2, style=pg.QtCore.Qt.DashLine),
            )
            self._phase_pvlines.append(line)

    def _redraw_all_vlines(self):
        for ch in range(1, 5):
            for line in self._bmode_vlines[ch]:
                self._bmode_plots[ch].removeItem(line)
            self._bmode_vlines[ch].clear()
            for m in self._phase_markers:
                line = self._bmode_plots[ch].addLine(
                    x=m,
                    pen=pg.mkPen("#FFFF00", width=1.2, style=pg.QtCore.Qt.DashLine),
                )
                self._bmode_vlines[ch].append(line)

    # ── 力矩计算 & 显示 ───────────────────────────────────────────────────────

    def _recompute_torque(self):
        """从当前相位 + 参数重新计算力矩并刷新显示。"""
        if self._n_frames == 0:
            return

        self._phase_arr = self._compute_phase()

        if self._lut is None:
            self._torque_arr = np.zeros(self._n_frames, dtype=np.float32)
            self._update_torque_display()
            return

        weight_factor = float(self._spin_weight.value()) / REFERENCE_WEIGHT
        phase_offset  = float(self._spin_offset.value())
        alpha         = float(self._spin_alpha.value())

        torque_arr      = np.zeros(self._n_frames, dtype=np.float32)
        torque_filtered = 0.0

        for i in range(self._n_frames):
            phase_corrected = (float(self._phase_arr[i]) + phase_offset) % 100.0
            torque_raw      = self._lut.get_torque(phase_corrected) * weight_factor
            torque_filtered = alpha * torque_raw + (1.0 - alpha) * torque_filtered
            torque_arr[i]   = torque_filtered

        self._torque_arr = torque_arr
        self._update_torque_display()

    def _update_torque_display(self):
        x = np.arange(self._n_frames)
        self._torque_curve.setData(x, self._torque_arr)

    # ── 保存 ─────────────────────────────────────────────────────────────────

    def _on_save(self):
        if not self._raw_frames:
            QMessageBox.warning(self, "错误", "请先加载超声 CSV")
            return
        if len(self._phase_markers) < 2:
            ret = QMessageBox.question(
                self, "相位边界不足",
                "当前相位边界少于 2 个，相位列将全为 0。\n是否继续保存？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return

        # 用当前系统时间作为文件名前缀
        from datetime import datetime
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = DATA_SAVE_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # 预检：是否有文件会被覆盖
        selected_chs = [ch for ch, cb in self._ch_checks.items() if cb.isChecked()]
        if not selected_chs:
            QMessageBox.warning(self, "未选通道", "请至少勾选一个通道")
            return

        existing = [f"{ts_tag}_{ch}.csv" for ch in selected_chs
                    if (out_dir / f"{ts_tag}_{ch}.csv").exists()]
        if existing:
            ret = QMessageBox.question(
                self, "文件已存在",
                f"以下文件已存在，保存将覆盖：\n" + "\n".join(existing) + "\n\n是否继续？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return

        self._lbl_status.setText("正在保存…")
        QApplication.processEvents()

        self._phase_arr  = self._compute_phase()
        self._recompute_torque()

        depth_cols = [f"d{i}" for i in range(ROI_N)]
        cols       = depth_cols + ["torque", "phase"]

        saved_paths = []
        for ch in selected_chs:
            raw_roi = self._raw_frames[ch][:, ROI_START:ROI_END]
            n       = min(len(raw_roi), len(self._torque_arr), len(self._phase_arr))
            rows    = np.hstack([
                raw_roi[:n],
                self._torque_arr[:n].reshape(-1, 1),
                self._phase_arr[:n].reshape(-1, 1),
            ])
            out_path = out_dir / f"{ts_tag}_{ch}.csv"
            pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
            saved_paths.append(out_path.name)

        QMessageBox.information(
            self, "保存成功",
            f"已保存到 {out_dir}:\n\n" + "\n".join(saved_paths),
        )
        self._lbl_status.setText(f"已保存 → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="步态相位 & 力矩标注工具")
    parser.add_argument("--file", default="", help="超声 CSV 路径（可选）")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = GaitLabeler(csv_file=args.file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
