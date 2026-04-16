"""
XSens Awinda 无线 IMU 采集进程
由 main_capture.py 通过 subprocess 启动，也可单独运行。

CSV 输出格式（每行一个采样）：
    timestamp, device_id,
    acc_x, acc_y, acc_z,
    gyr_x, gyr_y, gyr_z,
    mag_x, mag_y, mag_z,
    roll, pitch, yaw

运行：
    # 单独运行（自动命名输出目录）
    python capture_imu.py

    # 由 main_capture.py 调用（指定目录和文件名标签）
    python capture_imu.py --output-dir ./data/20260414_153022 --session-tag 20260414_153022

    # 定时采集
    python capture_imu.py --duration 60
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import xsensdeviceapi as xda

RADIO_CHANNEL  = 25   # 无线信道，可改 11~25（与 MTw 设备保持一致）
UPDATE_RATE    = 60   # 采样率 Hz
WAIT_FOR_MTW   = 15   # 等待 MTw 连接的超时秒数
STABLE_WAIT    = 3.0  # 发现第一个设备后，再等待这么多秒让其他设备也连上

CSV_HEADER = [
    "timestamp", "device_id",
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z",
    "roll", "pitch", "yaw",
]


# ── 回调：挂在每个 MTw 子设备上，缓冲数据包 ──────────────────────────────
class MtwCallback(xda.XsCallback):
    def __init__(self, max_buffer=10):
        xda.XsCallback.__init__(self)
        self._buf = []
        self._lock = Lock()
        self._max = max_buffer

    def getAllPackets(self) -> list:
        with self._lock:
            pkts = self._buf.copy()
            self._buf.clear()
            return pkts

    def onLiveDataAvailable(self, dev, packet):
        with self._lock:
            while len(self._buf) >= self._max:
                self._buf.pop(0)
            self._buf.append((dev.deviceId().toXsString(), xda.XsDataPacket(packet)))


def packet_to_row(ts: float, did_str: str, packet: xda.XsDataPacket) -> list:
    """将数据包解析为 CSV 一行"""
    acc_x = acc_y = acc_z = ""
    gyr_x = gyr_y = gyr_z = ""
    mag_x = mag_y = mag_z = ""
    roll = pitch = yaw = ""

    if packet.containsCalibratedData():
        acc = packet.calibratedAcceleration()
        acc_x, acc_y, acc_z = acc[0], acc[1], acc[2]
        gyr = packet.calibratedGyroscopeData()
        gyr_x, gyr_y, gyr_z = gyr[0], gyr[1], gyr[2]
        mag = packet.calibratedMagneticField()
        mag_x, mag_y, mag_z = mag[0], mag[1], mag[2]

    if packet.containsOrientation():
        euler = packet.orientationEuler()
        roll, pitch, yaw = euler.x(), euler.y(), euler.z()

    return [
        f"{ts:.6f}", did_str,
        acc_x, acc_y, acc_z,
        gyr_x, gyr_y, gyr_z,
        mag_x, mag_y, mag_z,
        roll, pitch, yaw,
    ]


def main():
    parser = argparse.ArgumentParser(description="XSens IMU 采集进程")
    parser.add_argument("--output-dir",  type=str, default="./data",
                        help="CSV 输出目录（默认 ./data）")
    parser.add_argument("--session-tag", type=str, default=None,
                        help="文件名时间标签，不填则自动生成")
    parser.add_argument("--duration",    type=float, default=0,
                        help="采集时长（秒），0 表示持续到 Ctrl+C（默认 0）")
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    session_tag = args.session_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"imu_{session_tag}.csv"

    print("=== XSens IMU 采集进程 ===")
    print(f"  输出文件: {csv_path}")
    print(f"  采集时长: {'持续' if args.duration == 0 else f'{args.duration:.0f} 秒'}\n")

    control       = xda.XsControl_construct()
    master_device = None
    master_port   = xda.XsPortInfo()
    callback      = None
    count         = 0
    csv_file      = None

    try:
        # ── 1. 扫描端口，找 WirelessMaster（Dongle）──────────────────────
        print("扫描端口，寻找 Awinda Dongle...")
        ports = xda.XsScanner_scanPorts()
        for i in range(ports.size()):
            p = ports[i]
            if p.deviceId().isWirelessMaster():
                master_port = p
                break

        if master_port.empty():
            raise RuntimeError("未找到 WirelessMaster（Awinda Dongle），请检查 USB 连接")

        print(f"使用 Dongle: {master_port.portName()}  ID={master_port.deviceId().toXsString()}")

        # ── 2. 打开端口 ──────────────────────────────────────────────────
        if not control.openPort(master_port.portName(), master_port.baudrate()):
            raise RuntimeError(f"无法打开端口 {master_port.portName()}")

        master_device = control.device(master_port.deviceId())
        assert master_device != 0

        # ── 3. 进入配置模式，开启无线电 ──────────────────────────────────
        if not master_device.gotoConfig():
            raise RuntimeError("无法进入配置模式")

        print(f"开启无线电（信道 {RADIO_CHANNEL}）...")
        if not master_device.enableRadio(RADIO_CHANNEL):
            raise RuntimeError(f"无法开启无线电，信道 {RADIO_CHANNEL}")

        # ── 4. 等待 MTw 设备连接 ─────────────────────────────────────────
        # 先等到至少出现 1 个设备，再等 STABLE_WAIT 秒让其他设备也连上
        print(f"等待 MTw 设备连接（最多 {WAIT_FOR_MTW} 秒）...")
        mtw_devices = []
        first_found_time = None
        deadline = time.time() + WAIT_FOR_MTW

        while time.time() < deadline:
            mtw_devices = master_device.children()
            if len(mtw_devices) > 0 and first_found_time is None:
                first_found_time = time.time()
                print(f"\n发现第一个 MTw，等待 {STABLE_WAIT:.0f}s 让其他设备也连上...")
            # 在第一个设备出现后，再等 STABLE_WAIT 秒
            if first_found_time and (time.time() - first_found_time) >= STABLE_WAIT:
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        print()

        if not mtw_devices:
            raise RuntimeError("超时：未发现任何 MTw 设备")

        print(f"共连接 {len(mtw_devices)} 个 MTw 设备")
        for mtw in mtw_devices:
            print(f"  MTw ID: {mtw.deviceId().toXsString()}")

        # ── 5. 配置采样率和输出 ───────────────────────────────────────────
        supported = master_device.supportedUpdateRates()
        rates = [supported[i] for i in range(supported.size())]
        rate  = UPDATE_RATE if UPDATE_RATE in rates else rates[-1]
        print(f"设置采样率: {rate} Hz")
        master_device.setUpdateRate(rate)

        cfg = xda.XsOutputConfigurationArray()
        cfg.push_back(xda.XsOutputConfiguration(xda.XDI_EulerAngles, rate))
        cfg.push_back(xda.XsOutputConfiguration(xda.XDI_Acceleration, rate))
        cfg.push_back(xda.XsOutputConfiguration(xda.XDI_RateOfTurn, rate))
        cfg.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, rate))

        callback = MtwCallback()
        for mtw in mtw_devices:
            mtw.addCallbackHandler(callback)
            if not mtw.setOutputConfiguration(cfg):
                print(f"  警告: {mtw.deviceId().toXsString()} 配置输出失败")

        # ── 6. 进入测量模式 ───────────────────────────────────────────────
        print("进入测量模式...")
        if not master_device.gotoMeasurement():
            raise RuntimeError("无法进入测量模式")

        # ── 7. 打开 CSV 文件 ──────────────────────────────────────────────
        csv_file = open(csv_path, "w", newline="", buffering=1)
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        print(f"\nCSV 已创建: {csv_path}")
        print("采集中（Ctrl+C 停止）...\n")

        # ── 8. 采集主循环 ─────────────────────────────────────────────────
        start_time = time.time()
        end_time   = start_time + args.duration if args.duration > 0 else None

        while True:
            if end_time and time.time() >= end_time:
                print(f"已达到采集时长 {args.duration:.0f}s，停止")
                break

            pkts = callback.getAllPackets()
            for did_str, packet in pkts:
                ts = time.time()
                count += 1
                writer.writerow(packet_to_row(ts, did_str, packet))

            if not pkts:
                time.sleep(0.001)
            else:
                # 每 100 行打印一次进度，避免刷屏
                if count % 100 == 0:
                    print(f"  [IMU] 已写入 {count} 行", flush=True)

    except KeyboardInterrupt:
        print(f"\n收到 Ctrl+C，停止采集")
    except RuntimeError as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if csv_file:
            csv_file.close()
        print(f"IMU 采集结束，共 {count} 行  →  {csv_path}")
        if master_device:
            try:
                master_device.gotoConfig()
                master_device.disableRadio()
            except Exception:
                pass
        if not master_port.empty():
            control.closePort(master_port.portName())
        control.close()


if __name__ == "__main__":
    main()
