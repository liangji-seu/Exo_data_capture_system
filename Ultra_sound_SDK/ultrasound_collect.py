"""
超声信号采集脚本（无 LSL 依赖版）
功能：自动搜索设备IP → 配置设备 → 开启设备 → 下发配置 → 开始采集 → 实时打印超声数据 → 停止采集

直接使用底层 Newsletter（C# SDK），不依赖 elonxiPy / pylsl / LSL。

运行方式：
    cd Ultra_sound_SDK

    # 自动搜索设备 IP（推荐）
    python ultrasound_collect.py --channels 1,2

    # 手动指定 IP
    python ultrasound_collect.py --device-ip 192.168.137.222 --channels 1,2

    # 定时采集 30 秒
    python ultrasound_collect.py --channels 1,2 --duration 30
"""

import argparse
import time
import threading

# zeroconf 是可选依赖（自动搜索功能），未安装时只能手动指定 --device-ip
try:
    from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

from pythonnet import load
load("coreclr")
import clr

clr.AddReference('System.Collections')
clr.AddReference("./无线/Elonxi_SDK")

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
#  全局统计变量
# ─────────────────────────────────────────────
total_packets    = 0   # 累计收到的超声数据包数
curr_pack_number = 0   # 当前超声包编号（由 RelData 回调更新）


# ─────────────────────────────────────────────
#  全局事件回调（C# SDK 直接回调，无 Qt 依赖）
# ─────────────────────────────────────────────

def on_notification(packet_type, message):
    """设备通知回调：连接、配置、采集状态等"""
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
    超声数据回调：每次硬件推送超声数据时触发
    :param ultrasonic_data_by_channel: dict，key=通道号(int)，value=波形列表(List[int[]])
    """
    global total_packets
    for ch, waveforms in ultrasonic_data_by_channel.items():
        for wf in waveforms:
            data = list(wf)
            total_packets += 1
            peak  = max(data) if data else 0
            first = data[:5]
            print(
                f"[超声] pack={curr_pack_number:>6d}  "
                f"ch={ch}  len={len(data):>4d}  "
                f"peak={peak:>6d}  first5={first}"
            )


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Elonxi 超声信号采集脚本（无 LSL 版）")
    parser.add_argument("--device-ip", type=str,   default=None,
                        help="设备 IP 地址（不填则自动搜索），例如 192.168.137.222")
    parser.add_argument("--port",      type=int,   default=1430,
                        help="通信端口（默认 1430）")
    parser.add_argument("--channels",  type=str,   default="1,2,3,4",
                        help="超声通道，多通道用逗号分隔，例如 '1,2,3,4'（默认 '1,2,3,4'）")
    parser.add_argument("--duration",  type=float, default=0,
                        help="采集时长（秒），0 表示持续采集直到 Ctrl+C（默认 0）")
    args = parser.parse_args()

    ult_channels = args.channels.strip()

    # ── IP 确定：手动指定 或 自动搜索 ───────────────
    device_ip = args.device_ip
    if not device_ip:
        device_ip = search_device(timeout=10.0)
        if not device_ip:
            print("错误：未找到设备，退出。请用 --device-ip 手动指定 IP 或确认设备已开机")
            return

    print("=" * 55)
    print("  Elonxi 超声信号采集（无 LSL 版）")
    print("=" * 55)
    print(f"  设备 IP   : {device_ip}")
    print(f"  端口      : {args.port}")
    print(f"  超声通道  : {ult_channels}")
    print(f"  采集时长  : {'持续' if args.duration == 0 else f'{args.duration} 秒'}")
    print("=" * 55)

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
    # Newsletter.configParam 签名（C# SDK，6 个参数）：
    #   configParam(ultr, emg, imu, inputMod, outMod, emgMod)
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
    print("  正在接收超声数据，按 Ctrl+C 停止...")
    print("─" * 55)

    # ── 采集主循环 ────────────────────────────────────
    try:
        if args.duration > 0:
            end_time = time.time() + args.duration
            while time.time() < end_time:
                remaining = end_time - time.time()
                print(f"  [剩余 {remaining:.1f}s]", end="\r", flush=True)
                time.sleep(0.1)
        else:
            while True:
                time.sleep(0.1)

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

    # ── 统计输出 ──────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  采集结束，共收到超声数据包: {total_packets} 包")
    print("=" * 55)


if __name__ == "__main__":
    main()
