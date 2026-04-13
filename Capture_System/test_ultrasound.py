"""
Elonxi A 型超声单独测试 - 终端打印
运行:  python test_ultrasound.py --device-ip 192.168.137.210
"""

import argparse
import signal
import sys
import time

from elonxi_reader import ElonxiReader


def main():
    parser = argparse.ArgumentParser(description="Elonxi A 型超声测试")
    parser.add_argument("--device-ip", type=str, required=True, help="设备 IP")
    parser.add_argument("--channels", type=int, nargs="+", default=[0], help="超声通道（默认 [0]）")
    args = parser.parse_args()

    print("=== Elonxi A 型超声测试 ===\n")

    reader = ElonxiReader(ult_channels=args.channels)
    reader.connect(args.device_ip)

    ult_ch_str = ",".join(str(c) for c in args.channels)
    print(f"配置超声通道: {ult_ch_str}")
    reader.config(ult_channel_str=ult_ch_str)

    reader.start_collection()

    running = True
    count = 0
    start_time = time.time()

    def on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    print("采集中（Ctrl+C 停止）...\n")

    try:
        while running:
            while reader.data_available():
                item = reader.get_data(timeout=0)
                if item is None:
                    break
                ts, ch, waveform = item
                count += 1
                peak = max(waveform) if waveform else 0
                mean_val = sum(waveform) / len(waveform) if waveform else 0
                print(
                    f"#{count:5d}  ch={ch}  len={len(waveform)}  "
                    f"peak={peak:6d}  mean={mean_val:8.1f}  "
                    f"first5={waveform[:5]}"
                )
            time.sleep(0.001)
    finally:
        elapsed = time.time() - start_time
        print(f"\n=== 停止 === 共 {count} 帧, {elapsed:.1f}s")
        reader.stop_collection()
        reader.disconnect()


if __name__ == "__main__":
    main()
