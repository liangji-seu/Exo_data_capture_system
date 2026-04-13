"""
主采集系统 - 同步采集 XSens IMU 和 Elonxi 超声数据

运行:
    python main_capture.py [--imu-rate 100] [--ult-channels 0 1] [--device-ip 192.168.x.x]
                           [--duration 60] [--output-dir ./data]

输出:
    data/
        imu_YYYYMMDD_HHMMSS.csv      XSens IMU 原始数据
        ult_YYYYMMDD_HHMMSS.csv      Elonxi 超声回波数据
"""

import argparse
import csv
import os
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

from elonxi_reader import ElonxiReader
from xsens_reader import XSensReader


# ──────────────────────────────────────────────
# CSV 写入器（后台线程）
# ──────────────────────────────────────────────

class CsvWriter:
    """异步 CSV 写入器，将数据从队列写入文件"""

    def __init__(self, filepath: Path, header: list[str]):
        self._filepath = filepath
        self._header = header
        self._rows: list[list] = []
        self._lock = threading.Lock()
        self._flush_interval = 2.0  # 每 2 秒刷新一次
        self._running = False
        self._thread: threading.Thread | None = None
        self._file = None
        self._writer = None

    def start(self):
        self._file = open(self._filepath, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._header)
        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def write(self, row: list):
        """线程安全写入一行"""
        with self._lock:
            self._rows.append(row)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._flush_now()
        if self._file:
            self._file.close()

    def _flush_loop(self):
        while self._running:
            time.sleep(self._flush_interval)
            self._flush_now()

    def _flush_now(self):
        with self._lock:
            rows = self._rows[:]
            self._rows.clear()
        if rows and self._writer:
            self._writer.writerows(rows)


# ──────────────────────────────────────────────
# 主采集类
# ──────────────────────────────────────────────

class CaptureSystem:
    def __init__(
        self,
        ult_channels: list[int],
        device_ip: str | None,
        imu_rate: int,
        duration: float | None,
        output_dir: Path,
    ):
        self.ult_channels = ult_channels
        self.device_ip = device_ip
        self.imu_rate = imu_rate
        self.duration = duration
        self.output_dir = output_dir

        self._imu_writer: CsvWriter | None = None
        self._ult_writer: CsvWriter | None = None
        self._ult_waveform_len: int | None = None  # 首次收到数据时确定

        self._elonxi = ElonxiReader(ult_channels=ult_channels)
        self._xsens = XSensReader(sample_rate=imu_rate)

        self._running = False
        self._imu_count = 0
        self._ult_count = 0

    # ------------------------------------------------------------------
    # IMU CSV 头（设备类型未知时先用最全的，后续根据实际数据动态创建）
    # ------------------------------------------------------------------

    @staticmethod
    def _imu_header() -> list[str]:
        return [
            "timestamp",
            # IMU 原始数据
            "acc_x", "acc_y", "acc_z",
            "gyr_x", "gyr_y", "gyr_z",
            "mag_x", "mag_y", "mag_z",
            # VRU/AHRS 姿态
            "quat_w", "quat_x", "quat_y", "quat_z",
            "roll", "pitch", "yaw",
            # GNSS
            "lat", "lon", "altitude",
            "vel_e", "vel_n", "vel_u",
        ]

    def _ult_header(self, n_points: int) -> list[str]:
        header = ["timestamp", "channel"]
        for ch in self.ult_channels:
            for i in range(n_points):
                header.append(f"ult_ch{ch}_pt{i}")
        return header

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── 连接 XSens IMU ──
        print("\n=== 初始化 XSens IMU ===")
        if not self._xsens.connect():
            print("[警告] XSens IMU 连接失败，将不采集 IMU 数据")
            xsens_ok = False
        else:
            xsens_ok = True
            imu_path = self.output_dir / f"imu_{ts_tag}.csv"
            self._imu_writer = CsvWriter(imu_path, self._imu_header())
            self._imu_writer.start()
            print(f"[XSens] IMU 数据将保存到: {imu_path}")

        # ── 连接 Elonxi 超声 ──
        print("\n=== 初始化 Elonxi 超声 ===")
        if self.device_ip is None:
            ips = self._elonxi.search_device(timeout=5.0)
            if ips:
                self.device_ip = ips[0]
                print(f"[Elonxi] 使用自动发现的 IP: {self.device_ip}")
            else:
                print("[警告] 未能发现 Elonxi 设备，将不采集超声数据")
                elonxi_ok = False
        else:
            elonxi_ok = True

        if self.device_ip:
            elonxi_ok = self._elonxi.connect(self.device_ip)
            if elonxi_ok:
                # 配置超声通道（如 "0,1"）
                ult_ch_str = ",".join(str(c) for c in self.ult_channels)
                self._elonxi.config(ult_channel_str=ult_ch_str)
        else:
            elonxi_ok = False

        if not xsens_ok and not elonxi_ok:
            print("[错误] 两个设备均连接失败，退出")
            return

        # ── 开始采集 ──
        print("\n=== 开始采集 ===")
        self._running = True

        if xsens_ok:
            self._xsens.start()
        if elonxi_ok:
            self._elonxi.start_collection()

        # 捕获 Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

        start_time = time.time()
        last_stat_time = start_time

        print(f"采集中... {'（持续 %.0f 秒）' % self.duration if self.duration else '（按 Ctrl+C 停止）'}\n")

        while self._running:
            now = time.time()

            # 检查持续时间
            if self.duration and (now - start_time) >= self.duration:
                print(f"\n[采集] 已达到设定时长 {self.duration:.0f}s，停止采集")
                break

            # ── 读取 IMU 数据 ──
            if xsens_ok:
                while self._xsens.data_available():
                    ts, imu_data = self._xsens.get_data(timeout=0)
                    if ts is not None:
                        self._write_imu(ts, imu_data)

            # ── 读取超声数据 ──
            if elonxi_ok:
                while self._elonxi.data_available():
                    item = self._elonxi.get_data(timeout=0)
                    if item is not None:
                        ts, ch, waveform = item
                        self._write_ult(ts, ch, waveform)

            # ── 状态输出 ──
            if now - last_stat_time >= 1.0:
                elapsed = now - start_time
                print(
                    f"\r已采集 {elapsed:.1f}s | IMU: {self._imu_count} 条 | 超声: {self._ult_count} 条",
                    end="", flush=True
                )
                last_stat_time = now

            time.sleep(0.001)  # 1ms 主循环间隔

        print("\n\n=== 停止采集 ===")
        self._stop()

    def _write_imu(self, ts: float, data: dict):
        """写入 IMU 一行 CSV"""
        row = [f"{ts:.6f}"]
        for col in self._imu_header()[1:]:  # 跳过 timestamp
            row.append(data.get(col, ""))
        self._imu_writer.write(row)
        self._imu_count += 1

    def _write_ult(self, ts: float, ch: int, waveform: list[int]):
        """写入超声一行 CSV"""
        # 首次收到数据时创建 CSV（确定波形长度）
        if self._ult_writer is None:
            n = len(waveform)
            self._ult_waveform_len = n
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            ult_path = self.output_dir / f"ult_{ts_tag}.csv"
            self._ult_writer = CsvWriter(ult_path, self._ult_header_flat(n))
            self._ult_writer.start()
            print(f"\n[Elonxi] 超声波形长度: {n} 点，数据保存到: {ult_path}")

        # 构建行：timestamp, channel, pt0, pt1, ..., ptN
        row = [f"{ts:.6f}", ch] + waveform
        self._ult_writer.write(row)
        self._ult_count += 1

    def _ult_header_flat(self, n_points: int) -> list[str]:
        """扁平化超声 CSV 列头：timestamp, channel, pt0, pt1, ..."""
        return ["timestamp", "channel"] + [f"pt{i}" for i in range(n_points)]

    def _stop(self):
        self._running = False
        self._xsens.stop()
        self._xsens.disconnect()
        self._elonxi.stop_collection()
        self._elonxi.disconnect()
        if self._imu_writer:
            self._imu_writer.stop()
        if self._ult_writer:
            self._ult_writer.stop()
        print(f"[完成] 共采集 IMU {self._imu_count} 条，超声 {self._ult_count} 条")

    def _signal_handler(self, sig, frame):
        print("\n[中断] 收到 Ctrl+C，正在停止...")
        self._running = False


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="IMU + 超声同步采集系统")
    parser.add_argument(
        "--imu-rate", type=int, default=100,
        help="XSens IMU 采样率 Hz（默认100）"
    )
    parser.add_argument(
        "--ult-channels", type=int, nargs="+", default=[0],
        help="Elonxi 超声通道列表，例如 --ult-channels 0 1"
    )
    parser.add_argument(
        "--device-ip", type=str, default=None,
        help="Elonxi 设备 IP（留空则自动搜索）"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="采集持续时间（秒），留空则手动 Ctrl+C 停止"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data",
        help="CSV 输出目录（默认 ./data）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    system = CaptureSystem(
        ult_channels=args.ult_channels,
        device_ip=args.device_ip,
        imu_rate=args.imu_rate,
        duration=args.duration,
        output_dir=Path(args.output_dir),
    )
    system.run()
