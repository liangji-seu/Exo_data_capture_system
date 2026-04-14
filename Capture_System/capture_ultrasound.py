"""
Elonxi 超声采集进程
由 main_capture.py 通过 subprocess 启动，也可单独运行。

CSV 输出格式（每行一整包）：
    timestamp, channel, pack_num, d0, d1, ..., d999

运行：
    # 单独运行（自动搜索 IP，默认通道 1,2,3,4）
    python capture_ultrasound.py

    # 手动指定 IP
    python capture_ultrasound.py --device-ip 192.168.137.222

    # 由 main_capture.py 调用（指定目录和文件名标签）
    python capture_ultrasound.py --device-ip 192.168.137.222 \
        --output-dir ./data/20260414_153022 --session-tag 20260414_153022

    # 定时采集
    python capture_ultrasound.py --duration 60
"""

import argparse
import csv
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

# zeroconf 可选（用于自动搜索 IP）
try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

from pythonnet import load
load("coreclr")
import clr

_SDK_DIR = Path(__file__).parent.parent / "Ultra_sound_SDK" / "无线" / "Elonxi_SDK"
clr.AddReference('System.Collections')
clr.AddReference(str(_SDK_DIR))

from Elonxi_SDK import Newsletter, GlobalEvents, PacketType


# ─────────────────────────────────────────────
#  mDNS 设备自动搜索
# ─────────────────────────────────────────────

def search_device(timeout: float = 10.0):
    if not ZEROCONF_AVAILABLE:
        print("[搜索] zeroconf 未安装，请用 --device-ip 手动指定 IP")
        print("       安装方法: pip install zeroconf")
        return None

    class _Listener(ServiceListener):
        def __init__(self):
            self.found_ips = []
            self._event = threading.Event()

        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info:
                ips = info.parsed_addresses()
                print(f"  发现设备: {name}  IP={ips}")
                self.found_ips.extend(ips)
                self._event.set()

        def update_service(self, zc, type_, name): pass
        def remove_service(self, zc, type_, name): pass

    print(f"[搜索] 正在局域网内搜索设备（最多 {timeout:.0f} 秒）...")
    zeroconf = Zeroconf()
    listener = _Listener()
    browser  = ServiceBrowser(zeroconf, "_http._udp.local.", listener)
    found    = listener._event.wait(timeout=timeout)
    browser.cancel()
    zeroconf.close()

    if found and listener.found_ips:
        ip = listener.found_ips[0]
        print(f"[搜索] 使用设备 IP: {ip}")
        return ip
    print("[搜索] 未发现设备")
    return None


# ─────────────────────────────────────────────
#  全局状态
# ─────────────────────────────────────────────
_total_packets   = 0
_curr_pack_num   = 0
_csv_writer      = None
_csv_lock        = threading.Lock()


# ─────────────────────────────────────────────
#  C# SDK 回调
# ─────────────────────────────────────────────

def _on_notification(packet_type, message):
    if packet_type == PacketType.DeviceConnection:
        print(f"[通知] 设备连接: {message}")
    elif packet_type == PacketType.Configuration:
        print(f"[通知] 配置结果: {message}")
    elif packet_type == PacketType.CollectionStatus:
        print(f"[通知] 采集状态: {message}")
    elif packet_type == PacketType.BatteryCapacity:
        print(f"[通知] 电池电量: {message}")
    elif packet_type == PacketType.IsDeviceOnline:
        print(f"[通知] 设备在线: {message}")


def _on_rel_data(is_ult, pack_number):
    global _curr_pack_num
    if is_ult:
        _curr_pack_num = pack_number


def _on_ultrasound_data(ultrasonic_data_by_channel):
    global _total_packets, _csv_writer
    recv_time = time.time()

    for ch, waveforms in ultrasonic_data_by_channel.items():
        for wf in waveforms:
            data = list(wf)
            _total_packets += 1
            peak = max(data) if data else 0
            print(f"[超声] pack={_curr_pack_num:>6d}  ch={ch}  "
                  f"len={len(data):>4d}  peak={peak:>6d}")

            if _csv_writer is not None:
                with _csv_lock:
                    _csv_writer.writerow(
                        [f"{recv_time:.6f}", ch, _curr_pack_num] + data
                    )


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Elonxi 超声采集进程")
    parser.add_argument("--device-ip",   type=str,   default=None,
                        help="设备 IP（不填则自动搜索）")
    parser.add_argument("--port",        type=int,   default=1430,
                        help="通信端口（默认 1430）")
    parser.add_argument("--channels",    type=str,   default="1,2,3,4",
                        help="超声通道，逗号分隔（默认 '1,2,3,4'）")
    parser.add_argument("--duration",    type=float, default=0,
                        help="采集时长（秒），0 表示持续到 Ctrl+C")
    parser.add_argument("--output-dir",  type=str,   default="./data",
                        help="CSV 输出目录（默认 ./data）")
    parser.add_argument("--session-tag", type=str,   default=None,
                        help="文件名时间标签，不填则自动生成")
    args = parser.parse_args()

    ult_channels = args.channels.strip()
    output_dir   = Path(args.output_dir)
    session_tag  = args.session_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"ultrasound_{session_tag}.csv"

    # ── 确定设备 IP ──────────────────────────────────────────────────────
    device_ip = args.device_ip
    if not device_ip:
        device_ip = search_device(timeout=10.0)
        if not device_ip:
            print("[错误] 未找到设备，退出。请用 --device-ip 手动指定", file=sys.stderr)
            sys.exit(1)

    print("=" * 55)
    print("  Elonxi 超声采集进程")
    print("=" * 55)
    print(f"  设备 IP   : {device_ip}")
    print(f"  超声通道  : {ult_channels}")
    print(f"  采集时长  : {'持续' if args.duration == 0 else f'{args.duration:.0f} 秒'}")
    print(f"  输出文件  : {csv_path.resolve()}")
    print("=" * 55)

    # ── 打开 CSV ─────────────────────────────────────────────────────────
    global _csv_writer
    csv_file = open(csv_path, "w", newline="", encoding="utf-8", buffering=1)
    _csv_writer = csv.writer(csv_file)
    header = ["timestamp", "channel", "pack_num"] + [f"d{i}" for i in range(1000)]
    _csv_writer.writerow(header)

    # ── 注册事件回调 ──────────────────────────────────────────────────────
    GlobalEvents.NotificationReceived     += _on_notification
    GlobalEvents.RealRealUltrDataReceived += _on_ultrasound_data
    GlobalEvents.RealRealRelDataReceived  += _on_rel_data

    # ── 创建连接并采集 ────────────────────────────────────────────────────
    newsletter = Newsletter(args.port, device_ip, args.port)
    newsletter.deviceSwitch(True)
    time.sleep(2)

    newsletter.configParam(ult_channels, "", "", 0, 0, False)
    time.sleep(1)

    newsletter.collectionSwitch(True)
    time.sleep(0.5)
    print("\n采集中（Ctrl+C 停止）...\n")

    try:
        if args.duration > 0:
            end_time = time.time() + args.duration
            while time.time() < end_time:
                print(f"  [剩余 {end_time - time.time():.1f}s  已收 {_total_packets} 包]",
                      end="\r", flush=True)
                time.sleep(0.1)
        else:
            while True:
                print(f"  [已收 {_total_packets} 包]", end="\r", flush=True)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止采集")

    newsletter.collectionSwitch(False)
    time.sleep(0.5)
    newsletter.deviceSwitch(False)

    _csv_writer = None
    csv_file.flush()
    csv_file.close()

    print(f"\n超声采集结束，共 {_total_packets} 包  →  {csv_path}")


if __name__ == "__main__":
    main()
