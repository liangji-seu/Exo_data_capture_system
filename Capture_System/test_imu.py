"""
XSens Awinda 无线 IMU 测试脚本
实时打印 IMU 数据到终端，并保存到 CSV 文件。

输出文件: ./data/imu_YYYYMMDD_HHMMSS.csv
每行一个采样，列：timestamp, device_id,
              acc_x, acc_y, acc_z,
              gyr_x, gyr_y, gyr_z,
              mag_x, mag_y, mag_z,
              roll, pitch, yaw

运行:
    python test_imu.py
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import xsensdeviceapi as xda

RADIO_CHANNEL = 25   # 无线信道，可改 11~25（与 MTw 设备保持一致）
UPDATE_RATE   = 60   # 采样率 Hz
WAIT_FOR_MTW  = 15   # 等待 MTw 连接的超时秒数
OUTPUT_DIR    = Path("./data")  # CSV 输出目录

CSV_HEADER = [
    "timestamp",
    "device_id",
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z",
    "roll", "pitch", "yaw",
]


# ── 回调：挂在每个 MTw 子设备上，缓冲数据包（同时保存 deviceId 字符串）──
class MtwCallback(xda.XsCallback):
    def __init__(self, max_buffer=10):
        xda.XsCallback.__init__(self)
        self._buf = []
        self._lock = Lock()
        self._max = max_buffer

    def packetAvailable(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def getNextPacket(self):
        with self._lock:
            return self._buf.pop(0)  # 返回 (did_str, XsDataPacket)

    def getAllPackets(self) -> list:
        with self._lock:
            pkts = self._buf.copy()
            self._buf.clear()
            return pkts

    def onLiveDataAvailable(self, dev, packet):
        with self._lock:
            while len(self._buf) >= self._max:
                self._buf.pop(0)
            # 同时保存设备 ID 字符串，方便多设备区分
            self._buf.append((dev.deviceId().toXsString(), xda.XsDataPacket(packet)))


def print_packet(did_str: str, packet: xda.XsDataPacket, count: int):
    parts = []

    if packet.containsCalibratedData():
        acc = packet.calibratedAcceleration()
        gyr = packet.calibratedGyroscopeData()
        mag = packet.calibratedMagneticField()
        parts.append(f"Acc[{acc[0]:7.3f} {acc[1]:7.3f} {acc[2]:7.3f}]m/s²")
        parts.append(f"Gyr[{gyr[0]:7.3f} {gyr[1]:7.3f} {gyr[2]:7.3f}]rad/s")
        parts.append(f"Mag[{mag[0]:6.2f} {mag[1]:6.2f} {mag[2]:6.2f}]")

    if packet.containsOrientation():
        euler = packet.orientationEuler()
        parts.append(f"Roll {euler.x():7.2f}° Pitch {euler.y():7.2f}° Yaw {euler.z():7.2f}°")

    if parts:
        line = f"#{count:5d} [{did_str}] " + "  |  ".join(parts)
        print(f"\r{line:<140}", end="", flush=True)


def packet_to_row(ts: float, did_str: str, packet: xda.XsDataPacket) -> list:
    """将一个数据包解析为 CSV 一行（与 CSV_HEADER 对应）"""
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
    print("=== XSens Awinda 无线 IMU 测试 ===\n")

    control = xda.XsControl_construct()
    assert control != 0

    ver = xda.XsVersion()
    xda.xdaVersion(ver)
    print(f"XDA 版本: {ver.toXsString()}\n")

    master_port   = xda.XsPortInfo()
    master_device = None
    callback      = None   # 单个共享回调，挂到所有 MTw 上

    try:
        # ── 1. 扫描端口，找 WirelessMaster（Dongle）──
        print("扫描端口，寻找 Awinda Dongle...")
        ports = xda.XsScanner_scanPorts()
        for i in range(ports.size()):
            p = ports[i]
            print(f"  发现: {p.portName()}  ID={p.deviceId().toXsString()}"
                  f"  isWirelessMaster={p.deviceId().isWirelessMaster()}")
            if p.deviceId().isWirelessMaster():
                master_port = p
                break

        if master_port.empty():
            raise RuntimeError("未找到 WirelessMaster（Awinda Dongle），请检查 USB 连接")

        print(f"\n使用 Dongle: {master_port.portName()}  ID={master_port.deviceId().toXsString()}")

        # ── 2. 打开 Dongle 端口 ──
        if not control.openPort(master_port.portName(), master_port.baudrate()):
            raise RuntimeError(f"无法打开端口 {master_port.portName()}")

        master_device = control.device(master_port.deviceId())
        assert master_device != 0
        print(f"产品型号: {master_device.productCode()}")

        # ── 3. 进入配置模式，开启无线电 ──
        if not master_device.gotoConfig():
            raise RuntimeError("无法进入配置模式")

        print(f"\n开启无线电（信道 {RADIO_CHANNEL}）...")
        if not master_device.enableRadio(RADIO_CHANNEL):
            raise RuntimeError(f"无法开启无线电，信道 {RADIO_CHANNEL}")

        # ── 4. 轮询 children() 等待 MTw 出现（参考旧代码的可靠方式）──
        print(f"等待 MTw 设备连接（最多 {WAIT_FOR_MTW} 秒，请确认 MTw 已开机）...")
        mtw_devices = []
        deadline = time.time() + WAIT_FOR_MTW
        while time.time() < deadline:
            mtw_devices = master_device.children()
            if len(mtw_devices) > 0:
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        print()

        if len(mtw_devices) == 0:
            raise RuntimeError("超时：未发现任何 MTw 设备，请确认设备已开机")

        print(f"共连接 {len(mtw_devices)} 个 MTw 设备")
        for mtw in mtw_devices:
            print(f"  MTw ID: {mtw.deviceId().toXsString()}")

        # ── 5. 配置输出并挂回调 ──
        # 设置采样率
        supported = master_device.supportedUpdateRates()
        rates = [supported[i] for i in range(supported.size())]
        print(f"支持的采样率: {rates}")
        rate = UPDATE_RATE if UPDATE_RATE in rates else rates[-1]
        print(f"设置采样率: {rate} Hz")
        master_device.setUpdateRate(rate)

        # 输出配置
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

        # ── 6. 进入测量模式 ──
        print("\n进入测量模式...")
        if not master_device.gotoMeasurement():
            raise RuntimeError("无法进入测量模式")

        print("开始接收数据（按 Ctrl+C 停止）\n")
        print("-" * 80)

        # ── 建立 CSV 文件 ──
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"imu_{ts_tag}.csv"
        csv_file = open(csv_path, "w", newline="", buffering=1)  # line-buffered
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        print(f"CSV 保存到: {csv_path}\n")

        count = 0
        try:
            while True:
                pkts = callback.getAllPackets()
                for did_str, packet in pkts:
                    ts = time.time()
                    count += 1
                    print_packet(did_str, packet, count)
                    writer.writerow(packet_to_row(ts, did_str, packet))
                if not pkts:
                    time.sleep(0.001)
        finally:
            csv_file.close()
            print(f"\n\nCSV 已保存: {csv_path}  （共 {count} 行）")

    except KeyboardInterrupt:
        print(f"\n\n已停止，共接收 {count} 个数据包")
    except RuntimeError as e:
        print(f"\n错误: {e}")
        sys.exit(1)
    finally:
        print("\n正在清理...")
        if master_device:
            try:
                master_device.gotoConfig()
                master_device.disableRadio()
            except Exception:
                pass
        if not master_port.empty():
            control.closePort(master_port.portName())
        control.close()
        print("已关闭连接")


if __name__ == "__main__":
    main()
