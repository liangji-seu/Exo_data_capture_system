"""
最小化超声测试 - 参照供应商 UI.py 的调用方式
直接使用 Elonxi_SDK.dll，不依赖 PyQt5 / LSL

运行:
    cd Ultra_sound_SDK
    python test_sdk_minimal.py --device-ip 192.168.137.210
"""

import argparse
import time

from pythonnet import load
load("coreclr")
import clr

clr.AddReference('System.Collections')
clr.AddReference("./无线/Elonxi_SDK")

from Elonxi_SDK import Newsletter, GlobalEvents, PacketType


# ── 回调函数 ──

def on_notification(packet_type, message):
    print(f"[通知] type={packet_type}, msg={message}")


def on_ultrasound_data(ultrasonic_data_by_channel):
    print(f"[超声] 收到数据, 通道数: {len(ultrasonic_data_by_channel)}")
    for ch, waveforms in ultrasonic_data_by_channel.items():
        for wf in waveforms:
            data = list(wf)
            print(f"  ch={ch}  len={len(data)}  first5={data[:5]}  peak={max(data) if data else 0}")


def on_emg_data(emg_data_by_channel):
    print(f"[EMG] 收到数据, 通道数: {len(emg_data_by_channel)}")


def on_imu_data(imu_data):
    print(f"[IMU] 收到数据")


def on_rel_data(is_ult, pack_number):
    print(f"[包编号] isUlt={is_ult}, packNumber={pack_number}")


# 和 UI.py 中 sendData_Sig 一样的通用回调
def on_send_data(channel, data_list, is_emg, pack_num):
    tag = "EMG" if is_emg else "超声"
    print(f"[{tag}] ch={channel}  pack={pack_num}  len={len(data_list)}  first5={data_list[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Elonxi SDK 最小化超声测试")
    parser.add_argument("--device-ip", type=str, required=True, help="设备 IP")
    parser.add_argument("--port", type=int, default=1430, help="端口（默认 1430，与供应商 UI 一致）")
    parser.add_argument("--ult-channels", type=str, default="0", help="超声通道（默认 '0'）")
    args = parser.parse_args()

    # ── 注册所有全局事件 ──
    print("注册事件回调...")
    GlobalEvents.NotificationReceived += on_notification
    GlobalEvents.RealRealUltrDataReceived += on_ultrasound_data
    GlobalEvents.RealRealEMGReceived += on_emg_data
    GlobalEvents.RealReaIMUReceived += on_imu_data
    GlobalEvents.RealRealRelDataReceived += on_rel_data

    # ── 创建连接（参照 UI.py: 本地端口和设备端口相同）──
    print(f"\n连接设备 {args.device_ip}:{args.port} (本地端口 {args.port})...")
    newsletter = Newsletter(args.port, args.device_ip, args.port)

    # ── 开启设备 ──
    print("发送 deviceSwitch(True)...")
    newsletter.deviceSwitch(True)
    time.sleep(2)

    # ── 下发配置（关键：ultr 填通道，emg 留空，第7个参数=20 参照 elonxiPy_1.py）──
    print(f"发送 configParam(ultr='{args.ult_channels}', emg='', imu='', 0, 0, False, 20)...")
    newsletter.configParam(args.ult_channels, "", "", 0, 0, False)
    time.sleep(1)

    # ── 开始采集 ──
    print("发送 collectionSwitch(True)...")
    newsletter.collectionSwitch(True)
    time.sleep(0.5)

    print("\n=== 等待超声数据（Ctrl+C 停止）===\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n=== 停止 ===")

    newsletter.collectionSwitch(False)
    time.sleep(0.5)
    newsletter.deviceSwitch(False)
    print("已断开")


if __name__ == "__main__":
    main()
