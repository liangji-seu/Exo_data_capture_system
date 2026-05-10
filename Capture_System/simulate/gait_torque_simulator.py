#!/usr/bin/env python3
"""
gait_torque_simulator.py
========================
从 imu_xxx.csv 中读取左腿 IMU（尾号 260E）数据，
在界面上显示 gyr_z 信号，让用户框选多段行走区间，
对每段独立计算步态相位，再通过 hip_torque_lut.csv 样条
计算髋关节力矩，最终输出 torque.csv 到 imu 文件同级目录。

算法完全移植自 gait_verify_manager/src/gait_verify_node.cpp。
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QDoubleSpinBox,
        QSpinBox, QMessageBox, QGroupBox,
    )
except ImportError:
    from PyQt5.QtCore import Qt, QRectF
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QDoubleSpinBox,
        QSpinBox, QMessageBox, QGroupBox,
    )

import pyqtgraph as pg

pg.setConfigOptions(antialias=True, background="w", foreground="k")

# ── 常量（与 gait_verify_node.hpp 保持一致）────────────────────────────────
LEFT_IMU_SUFFIX      = "260E"
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

LUT_PATH = Path(__file__).parent.parent / "gait_verify_manager" / "src" / "hip_torque_lut.csv"


# ── 步态相位检测器（直接移植 KneeGaitDetector）────────────────────────────

class KneeGaitDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_velocity_   = 0.0
        self.phase_           = 0.0
        self.last_peak_time_  = 0.0
        self.time_            = 0.0
        self.stride_period_   = DEFAULT_STRIDE_PERIOD
        self.in_swing_        = False
        self.last_motion_time_= 0.0

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
            # smootherstep: 6x^5 - 15x^4 + 10x^3
            self.phase_ = lp * lp * lp * (lp * (lp * 6.0 - 15.0) + 10.0)

        self.prev_velocity_ = angular_velocity_z
        return self.phase_


# ── 髋关节力矩查找表（移植 HipTorqueLUT）─────────────────────────────────

class HipTorqueLUT:
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path, header=None)
        self._data = df.iloc[:, 1].values.astype(float)   # 第二列是力矩值
        assert len(self._data) == 1001, f"LUT 应有 1001 行，实际 {len(self._data)}"

    def get_torque(self, phase_percent: float) -> float:
        phase_percent = max(0.0, min(100.0, phase_percent))
        idx = phase_percent * 10.0
        i0  = int(idx)
        i1  = min(i0 + 1, len(self._data) - 1)
        t   = idx - i0
        return self._data[i0] + t * (self._data[i1] - self._data[i0])


# ── 主窗口 ────────────────────────────────────────────────────────────────

class GaitTorqueSimulator(QWidget):
    def __init__(self, imu_path: str = ""):
        super().__init__()
        self.setWindowTitle("步态相位 & 髋关节力矩模拟器")
        self.resize(1400, 900)

        self._imu_df:   pd.DataFrame = None
        self._left_df:  pd.DataFrame = None   # 仅左腿 IMU 行
        self._imu_path: Path = None

        # 选区列表：每项为 (idx_start, idx_end) 对应 _left_df 的行索引
        self._segments: list[tuple[int, int]] = []
        self._seg_regions: list[pg.LinearRegionItem] = []  # 图上的选区控件

        self._lut = HipTorqueLUT(LUT_PATH)

        self._build_ui()

        if imu_path:
            self._load_file(imu_path)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6); root.setContentsMargins(8, 8, 8, 8)

        # 第一行：文件 + 参数
        row1 = QHBoxLayout()
        btn_open = QPushButton("打开 IMU CSV"); btn_open.setFixedWidth(120)
        btn_open.clicked.connect(self._on_open)
        self._lbl_file = QLabel("未选择文件"); self._lbl_file.setStyleSheet("color:gray")
        row1.addWidget(btn_open); row1.addWidget(self._lbl_file, stretch=1)

        grp = QGroupBox("参数")
        grp.setFixedHeight(60)
        pg_ly = QHBoxLayout(grp); pg_ly.setContentsMargins(6, 4, 6, 4); pg_ly.setSpacing(8)

        pg_ly.addWidget(QLabel("体重(kg):"))
        self._spin_weight = QDoubleSpinBox()
        self._spin_weight.setRange(30, 150); self._spin_weight.setValue(72); self._spin_weight.setFixedWidth(65)
        pg_ly.addWidget(self._spin_weight)

        pg_ly.addWidget(QLabel("相位偏移(%):"))
        self._spin_offset = QDoubleSpinBox()
        self._spin_offset.setRange(0, 100); self._spin_offset.setValue(DEFAULT_PHASE_OFFSET); self._spin_offset.setFixedWidth(65)
        pg_ly.addWidget(self._spin_offset)

        pg_ly.addWidget(QLabel("滤波α:"))
        self._spin_alpha = QDoubleSpinBox()
        self._spin_alpha.setRange(0.01, 1.0); self._spin_alpha.setValue(DEFAULT_FILTER_ALPHA)
        self._spin_alpha.setDecimals(2); self._spin_alpha.setFixedWidth(60)
        pg_ly.addWidget(self._spin_alpha)

        row1.addWidget(grp)
        root.addLayout(row1)

        # 第二行：操作按钮
        row2 = QHBoxLayout()
        self._btn_add_seg = QPushButton("➕ 添加选区"); self._btn_add_seg.setEnabled(False)
        self._btn_add_seg.clicked.connect(self._add_segment)
        self._btn_clear_seg = QPushButton("🗑 清除所有选区")
        self._btn_clear_seg.clicked.connect(self._clear_segments)
        self._btn_compute = QPushButton("⚡ 计算并导出 torque.csv"); self._btn_compute.setEnabled(False)
        self._btn_compute.setStyleSheet("font-weight:bold; color:#006600")
        self._btn_compute.clicked.connect(self._compute_and_export)
        self._lbl_status = QLabel("")
        row2.addWidget(self._btn_add_seg); row2.addWidget(self._btn_clear_seg)
        row2.addSpacing(20); row2.addWidget(self._btn_compute)
        row2.addWidget(self._lbl_status, stretch=1)
        root.addLayout(row2)

        # 图表区：上图 gyr_z，下图 相位 + 力矩
        self._pw_gyro = pg.PlotWidget(title="左腿 IMU gyr_z（拖动选区边界调整范围）")
        self._pw_gyro.setLabel("left", "gyr_z (rad/s)")
        self._pw_gyro.setLabel("bottom", "样本索引")
        self._pw_gyro.showGrid(x=True, y=True, alpha=0.3)
        self._curve_gyro = self._pw_gyro.plot(pen=pg.mkPen("#2196F3", width=1.2))
        root.addWidget(self._pw_gyro, stretch=2)

        self._pw_result = pg.PlotWidget(title="步态相位 & 髋关节力矩（各选区独立计算）")
        self._pw_result.setLabel("left", "相位(%) / 力矩(Nm)")
        self._pw_result.setLabel("bottom", "样本索引")
        self._pw_result.showGrid(x=True, y=True, alpha=0.3)
        self._pw_result.addLegend(offset=(10, 10))
        # 占位曲线（仅用于图例）
        self._curve_phase_legend  = self._pw_result.plot(pen=pg.mkPen("#FF5722", width=1.5), name="步态相位(%)")
        self._curve_torque_legend = self._pw_result.plot(pen=pg.mkPen("#4CAF50", width=1.5), name="髋关节力矩(Nm)")
        self._result_curves: list = []   # 动态添加的每段曲线
        root.addWidget(self._pw_result, stretch=2)

        # 链接两图 x 轴
        self._pw_result.setXLink(self._pw_gyro)

    # ── 文件加载 ──────────────────────────────────────────────────────────
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 IMU CSV", str(Path.cwd()), "CSV (*.csv)")
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e)); return

        if "device_id" not in df.columns or "gyr_z" not in df.columns:
            QMessageBox.critical(self, "格式错误", "缺少 device_id 或 gyr_z 列"); return

        # 筛选左腿 IMU（尾号 260E，大小写不敏感）
        mask = df["device_id"].astype(str).str.upper().str.contains(LEFT_IMU_SUFFIX.upper())
        left_df = df[mask].reset_index(drop=True)
        if left_df.empty:
            QMessageBox.critical(self, "未找到设备", f"未找到尾号含 {LEFT_IMU_SUFFIX} 的 IMU 数据"); return

        self._imu_df  = df
        self._left_df = left_df
        self._imu_path = Path(path)
        self._lbl_file.setText(f"{Path(path).name}  ({len(left_df)} 帧，左腿 IMU)")
        self._lbl_file.setStyleSheet("color:black")

        self._clear_segments()
        self._curve_gyro.setData(left_df["gyr_z"].values)
        self._btn_add_seg.setEnabled(True)
        self._btn_compute.setEnabled(False)
        self._lbl_status.setText("请添加行走区间选区，然后点击计算")

    # ── 选区管理 ──────────────────────────────────────────────────────────
    def _add_segment(self):
        if self._left_df is None: return
        n = len(self._left_df)
        # 默认放在中间 1/4 ~ 3/4
        lo = int(n * 0.25); hi = int(n * 0.75)
        # 如果已有选区，新选区放在最后一个右侧
        if self._seg_regions:
            last_hi = int(self._seg_regions[-1].getRegion()[1])
            lo = min(last_hi + 10, n - 100)
            hi = min(lo + max(100, n // 8), n - 1)

        region = pg.LinearRegionItem(
            values=[lo, hi],
            brush=pg.mkBrush(QColor(33, 150, 243, 40)),
            pen=pg.mkPen("#2196F3", width=1.5),
            movable=True,
        )
        region.sigRegionChangeFinished.connect(self._on_region_changed)
        self._pw_gyro.addItem(region)
        self._seg_regions.append(region)
        self._segments.append((lo, hi))
        self._btn_compute.setEnabled(True)
        self._lbl_status.setText(f"已添加 {len(self._seg_regions)} 个选区，可拖动边界调整")

    def _on_region_changed(self):
        # 同步 _segments 列表
        self._segments = []
        for r in self._seg_regions:
            lo, hi = r.getRegion()
            self._segments.append((int(lo), int(hi)))

    def _clear_segments(self):
        for r in self._seg_regions:
            self._pw_gyro.removeItem(r)
        self._seg_regions.clear()
        self._segments.clear()
        self._btn_compute.setEnabled(False)
        self._lbl_status.setText("选区已清除")
        self._clear_result_curves()

    def _clear_result_curves(self):
        for c in self._result_curves:
            self._pw_result.removeItem(c)
        self._result_curves.clear()
        self._curve_phase_legend.setData([], [])
        self._curve_torque_legend.setData([], [])

    # ── 计算 & 导出 ───────────────────────────────────────────────────────
    def _compute_and_export(self):
        if self._left_df is None or not self._segments:
            return

        n       = len(self._left_df)
        gyr_z   = self._left_df["gyr_z"].values.astype(float)
        ts      = self._left_df["timestamp"].values.astype(float)

        weight_factor = float(self._spin_weight.value()) / REFERENCE_WEIGHT
        phase_offset  = float(self._spin_offset.value())
        alpha         = float(self._spin_alpha.value())

        # 输出数组（全帧，未覆盖的行填 NaN）
        phase_out  = np.full(n, np.nan)
        torque_out = np.full(n, np.nan)

        # 清除上次结果曲线
        self._clear_result_curves()

        total_valid = 0
        for seg_lo, seg_hi in self._segments:
            lo = max(0, min(seg_lo, n - 1))
            hi = max(lo + 1, min(seg_hi, n))

            detector = KneeGaitDetector()
            torque_filtered = 0.0

            seg_x      = np.arange(lo, hi)
            seg_phase  = np.zeros(hi - lo)
            seg_torque = np.zeros(hi - lo)

            for idx, i in enumerate(range(lo, hi)):
                if i == lo:
                    dt = DT_CLAMP_DEFAULT
                else:
                    dt = ts[i] - ts[i - 1]
                    if dt <= 0 or dt > DT_CLAMP_MAX:
                        dt = DT_CLAMP_DEFAULT

                # 左腿用负 gyr_z（与 C++ 一致）
                phase = detector.update(-gyr_z[i], dt)

                phase_corrected = (phase * 100.0 + phase_offset) % 100.0
                torque_raw      = self._lut.get_torque(phase_corrected) * weight_factor
                torque_filtered = alpha * torque_raw + (1.0 - alpha) * torque_filtered

                seg_phase[idx]  = phase * 100.0
                seg_torque[idx] = torque_filtered

                phase_out[i]  = phase * 100.0
                torque_out[i] = torque_filtered

            total_valid += hi - lo

            # 每段独立画曲线，选区之间自然断开
            c_phase = self._pw_result.plot(
                seg_x, seg_phase,
                pen=pg.mkPen("#FF5722", width=1.5))
            c_torque = self._pw_result.plot(
                seg_x, seg_torque,
                pen=pg.mkPen("#4CAF50", width=1.5))
            self._result_curves.extend([c_phase, c_torque])

        # 导出 torque.csv（只保留有值的行，供调试用）
        valid_mask = ~np.isnan(phase_out)
        out_df = pd.DataFrame({
            "timestamp":     ts[valid_mask],
            "phase_pct":     phase_out[valid_mask],
            "hip_torque_Nm": torque_out[valid_mask],
        })
        out_path = self._imu_path.parent / "torque.csv"
        out_df.to_csv(out_path, index=False)

        # ── 按分区切割所有文件，写入 handle_data_XX/ ─────────────────────
        base_dir = self._imu_path.parent
        errors   = self._export_segments(base_dir, ts, phase_out, torque_out)

        msg = f"torque.csv 已保存\n有效帧数：{total_valid} / {n}\n\n"
        msg += f"已生成 {len(self._segments)} 个 handle_data_XX/ 数据集"
        if errors:
            msg += "\n\n警告：\n" + "\n".join(errors)
        self._lbl_status.setText(f"完成，{len(self._segments)} 个数据集已导出")
        QMessageBox.information(self, "导出完成", msg)

    def _export_segments(self, base_dir: Path, left_ts: np.ndarray,
                         phase_out: np.ndarray, torque_out: np.ndarray) -> list[str]:
        """
        按每个选区切割 imu / motor / ultrasound / torque，
        写入 handle_data_01/ handle_data_02/ ... 目录。
        返回错误信息列表（空表示全部成功）。
        """
        errors = []

        # 加载同级目录的 motor / ultrasound / imu（找第一个匹配文件）
        def load_csv(prefix: str):
            matches = sorted(base_dir.glob(f"{prefix}*.csv"))
            if not matches:
                errors.append(f"未找到 {prefix}*.csv，该文件将跳过")
                return None
            try:
                return pd.read_csv(matches[0])
            except Exception as e:
                errors.append(f"读取 {matches[0].name} 失败: {e}")
                return None

        motor_df     = load_csv("motor_")
        ultra_df     = load_csv("ultrasound_")
        imu_full_df  = load_csv("imu_")

        for seg_idx, (seg_lo, seg_hi) in enumerate(self._segments, start=1):
            lo = max(0, min(seg_lo, len(left_ts) - 1))
            hi = max(lo + 1, min(seg_hi, len(left_ts)))

            # 该分区的时间戳范围（来自左腿 IMU）
            t_start = float(left_ts[lo])
            t_end   = float(left_ts[hi - 1])

            seg_dir   = base_dir / f"handle_data_{seg_idx:02d}"
            input_dir = seg_dir / "input"
            label_dir = seg_dir / "label"
            input_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            # ── label: torque ──────────────────────────────────────────
            valid = ~np.isnan(phase_out)
            seg_valid = valid & (np.arange(len(left_ts)) >= lo) & (np.arange(len(left_ts)) < hi)
            torque_seg = pd.DataFrame({
                "timestamp":     left_ts[seg_valid],
                "phase_pct":     phase_out[seg_valid],
                "hip_torque_Nm": torque_out[seg_valid],
            })
            torque_seg.to_csv(label_dir / "torque.csv", index=False)

            # ── input: imu（全设备，按时间戳范围截取）─────────────────
            if imu_full_df is not None:
                ts_col = imu_full_df["timestamp"].values.astype(float)
                mask   = (ts_col >= t_start) & (ts_col <= t_end)
                imu_full_df[mask].to_csv(input_dir / "imu.csv", index=False)

            # ── input: motor ───────────────────────────────────────────
            if motor_df is not None:
                ts_col = motor_df["timestamp"].values.astype(float)
                mask   = (ts_col >= t_start) & (ts_col <= t_end)
                motor_df[mask].to_csv(input_dir / "motor.csv", index=False)

            # ── input: ultrasound ──────────────────────────────────────
            if ultra_df is not None:
                ts_col = ultra_df["timestamp"].values.astype(float)
                mask   = (ts_col >= t_start) & (ts_col <= t_end)
                ultra_df[mask].to_csv(input_dir / "ultrasound.csv", index=False)

        return errors


# ── 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="步态相位 & 髋关节力矩模拟器")
    parser.add_argument("--file", default="", help="imu_xxx.csv 路径")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = GaitTorqueSimulator(imu_path=args.file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
