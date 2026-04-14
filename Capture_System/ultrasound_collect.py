"""
超声信号采集脚本（带 CSV 数据记录）
功能：自动搜索设备IP → 配置设备 → 开启设备 → 下发配置 → 开始采集 → 记录数据到 CSV → 停止采集

CSV 格式（每行一整包，列区分采样点）：
    timestamp, channel, pack_num, d0, d1, d2, ..., d999
    - timestamp : 收到该包时的 Unix 时间戳（秒，浮点）
    - channel   : 超声通道号
    - pack_num  : 包编号
    - d0~d999   : 包内 1000 个采样值

依赖：pythonnet（必须），zeroconf（可选，用于自动搜索 IP）
运行方式：
    cd Capture_System

    # 自动搜索 IP，默认通道 1,2,3,4，自动命名 CSV
    python ultrasound_collect.py

    # 手动指定 IP
    python ultrasound_collect.py --device-ip 192.168.137.222

    # 指定通道 + 采集时长
    python ultrasound_collect.py --device-ip 192.168.137.222 --channels 1,2 --duration 30

    # 指定输出文件名
    python ultrasound_collect.py --output my_data.csv
"""

import argparse
import csv
import os
import time
import threading
from datetime import datetime

# zeroconf 是可选依赖（自动搜索功能），未安装时只能手动指定 --device-ip
try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

from pythonnet import load
load("coreclr")
import clr

# DLL 路径相对于本脚本所在目录的上级 Ultra_sound_SDK/无线/
_SDK_DIR = os.path.join(os.path.dirname(__file__), "..", "Ultra_sound_SDK", "无线", "Elonxi_SDK")
clr.AddReference('System.Collections')
clr.AddReference(_SDK_DIR)

from Elonxi_SDK import Newsletter, GlobalEvents, PacketType


# ─────────────────────────────────────────────
#  mDNS 设备自动搜索（需要 zeroconf 库）
# ─────────────────────────────────────────────

def search_device(timeout: float = 10.0):
    """
    通过 mDNS 自动搜索局域网内的 Elonxi 设备。
    :param timeout: 最长等待时间（秒），默认 10 秒
    :return: 找到的第一个设备 IP 字符串，超时或不可用返回 None
    """
    if not ZEROCONF_AVAILABLE:
        print("[搜索] zeroconf 库未安装，无法自动搜索，请用 --device-ip 手动指定 IP")
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

        def update_service(self, zc, type_, name):
            pass

        def remove_service(self, zc, type_, name):
            pass

    print(f"[搜索] 正在局域网内搜索设备（最多等待 {timeout:.0f} 秒）...")
    zeroconf = Zeroconf()
    listener = _Listener()
    browser  = ServiceBrowser(zeroconf, "_http._udp.local.", listener)

    found = listener._event.wait(timeout=timeout)
    browser.cancel()
    zeroconf.close()

    if found and listener.found_ips:
        ip = listener.found_ips[0]
        print(f"[搜索] 使用设备 IP: {ip}")
        return ip
    else:
        print("[搜索] 未发现设备，请检查设备是否开机并连接同一局域网")
        return None


# ─────────────────────────────────────────────
#  全局状态
# ─────────────────────────────────────────────
total_packets    = 0      # 累计收到的超声数据包数
total_samples    = 0      # 累计写入的采样点数
curr_pack_number = 0      # 当前超声包编号
_csv_writer      = None   # CSV writer 对象（全局，供回调写入）
_csv_lock        = threading.Lock()  # 写 CSV 的线程锁


# ─────────────────────────────────────────────
#  全局事件回调
# ─────────────────────────────────────────────

def on_notification(packet_type, message):
    """设备通知回调"""
    if packet_type == PacketType.DeviceConnection:
        print(f"[通知] 设备连接状态: {message}")
    elif packet_type == PacketType.Configuration:
        print(f"[通知] 配置下发结果: {message}")
    elif packet_type == PacketType.CollectionStatus:
        print(f"[通知] 采集状态: {message}")
    elif packet_type == PacketType.BatteryCapacity:
        print(f"[通知] 电池电量: {message}")
    elif packet_type == PacketType.IsDeviceOnline:
        print(f"[通知] 设备在线: {message}")
    else:
        print(f"[通知] type={packet_type}, msg={message}")


def on_rel_data(is_ult, pack_number):
    """包编号回调：更新当前超声包编号"""
    global curr_pack_number
    if is_ult:
        curr_pack_number = pack_number


def on_ultrasound_data(ultrasonic_data_by_channel):
    """
    超声数据回调：打印摘要 + 写入 CSV
    :param ultrasonic_data_by_channel: dict，key=通道号(int)，value=波形列表(List[int[]])
    """
    global total_packets, total_samples, _csv_writer

    recv_time = time.time()   # 收到此包的时间戳

    for ch, waveforms in ultrasonic_data_by_channel.items():
        for wf in waveforms:
            data = list(wf)
            total_packets += 1
            peak  = max(data) if data else 0
            print(
                f"[超声] pack={curr_pack_number:>6d}  "
                f"ch={ch}  len={len(data):>4d}  "
                f"peak={peak:>6d}  "
                f"time={recv_time:.3f}"
            )

            # 写入 CSV（每行一整包：timestamp, channel, pack_num, d0, d1, ..., d999）
            if _csv_writer is not None:
                with _csv_lock:
                    _csv_writer.writerow(
                        [f"{recv_time:.6f}", ch, curr_pack_number] + data
                    )
                total_samples += 1


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Elonxi 超声信号采集脚本（CSV 记录版）")
    parser.add_argument("--device-ip", type=str,   default=None,
                        help="设备 IP 地址（不填则自动搜索），例如 192.168.137.222")
    parser.add_argument("--port",      type=int,   default=1430,
                        help="通信端口（默认 1430）")
    parser.add_argument("--channels",  type=str,   default="1,2,3,4",
                        help="超声通道，多通道用逗号分隔（默认 '1,2,3,4'）")
    parser.add_argument("--duration",  type=float, default=0,
                        help="采集时长（秒），0 表示持续采集直到 Ctrl+C（默认 0）")
    parser.add_argument("--output",    type=str,   default=None,
                        help="CSV 输出文件名（默认按时间戳自动命名）")
    args = parser.parse_args()

    ult_channels = args.channels.strip()

    # ── IP 确定：手动指定 或 自动搜索 ───────────────
    device_ip = args.device_ip
    if not device_ip:
        device_ip = search_device(timeout=10.0)
        if not device_ip:
            print("错误：未找到设备，退出。请用 --device-ip 手动指定 IP 或确认设备已开机")
            return

    # ── CSV 文件名 ────────────────────────────────
    if args.output:
        csv_path = args.output
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"ultrasound_{timestamp_str}.csv"

    print("=" * 55)
    print("  Elonxi 超声信号采集（CSV 记录版）")
    print("=" * 55)
    print(f"  设备 IP   : {device_ip}")
    print(f"  端口      : {args.port}")
    print(f"  超声通道  : {ult_channels}")
    print(f"  采集时长  : {'持续' if args.duration == 0 else f'{args.duration} 秒'}")
    print(f"  CSV 输出  : {os.path.abspath(csv_path)}")
    print("=" * 55)

    # 打开 CSV，写表头：timestamp, channel, pack_num, d0, d1, ..., d999
    global _csv_writer
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(csv_file)
    header = ["timestamp", "channel", "pack_num"] + [f"d{i}" for i in range(1000)]
    _csv_writer.writerow(header)
    print(f"\nCSV 文件已创建: {csv_path}")

    # ── 步骤 1：注册全局事件回调 ──────────────────────
    print("\n[步骤 1] 注册事件回调...")
    GlobalEvents.NotificationReceived     += on_notification
    GlobalEvents.RealRealUltrDataReceived += on_ultrasound_data
    GlobalEvents.RealRealRelDataReceived  += on_rel_data
    print("  完成")

    # ── 步骤 2：创建通信对象 ──────────────────────────
    print(f"\n[步骤 2] 创建 Newsletter（本地端口={args.port}, 设备={device_ip}:{args.port}）...")
    newsletter = Newsletter(args.port, device_ip, args.port)
    print("  完成")

    # ── 步骤 3：开启设备（建立连接）──────────────────
    print("\n[步骤 3] 开启设备（deviceSwitch=True）...")
    newsletter.deviceSwitch(True)
    time.sleep(2)
    print("  完成")

    # ── 步骤 4：下发超声配置 ──────────────────────────
    print(f"\n[步骤 4] 下发配置（超声通道='{ult_channels}'，EMG/IMU 不启用）...")
    newsletter.configParam(ult_channels, "", "", 0, 0, False)
    time.sleep(1)
    print("  完成")

    # ── 步骤 5：开始采集 ──────────────────────────────
    print("\n[步骤 5] 开始采集（collectionSwitch=True）...")
    newsletter.collectionSwitch(True)
    time.sleep(0.5)
    print("  完成\n")

    print("─" * 55)
    print("  正在接收超声数据并写入 CSV，按 Ctrl+C 停止...")
    print("─" * 55)

    # ── 采集主循环 ────────────────────────────────────
    try:
        if args.duration > 0:
            end_time = time.time() + args.duration
            while time.time() < end_time:
                remaining = end_time - time.time()
                print(f"  [剩余 {remaining:.1f}s  已记录 {total_samples} 采样点]",
                      end="\r", flush=True)
                time.sleep(0.1)
        else:
            while True:
                print(f"  [已收 {total_packets} 包 / 已记录 {total_samples} 采样点]",
                      end="\r", flush=True)
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n  收到 Ctrl+C，停止采集...")

    # ── 步骤 6：停止采集 ──────────────────────────────
    print("\n[步骤 6] 停止采集（collectionSwitch=False）...")
    newsletter.collectionSwitch(False)
    time.sleep(0.5)
    print("  完成")

    # ── 步骤 7：断开设备连接 ──────────────────────────
    print("\n[步骤 7] 断开设备（deviceSwitch=False）...")
    newsletter.deviceSwitch(False)
    time.sleep(0.5)
    print("  完成")

    # ── 关闭 CSV ──────────────────────────────────────
    _csv_writer = None
    csv_file.flush()
    csv_file.close()

    # ── 统计输出 ──────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  采集结束")
    print(f"  共收到超声数据包 : {total_packets} 包")
    print(f"  共写入数据行         : {total_samples} 行（每行=一包1000点）")
    print(f"  CSV 文件路径     : {os.path.abspath(csv_path)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
