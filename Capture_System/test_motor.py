#!/usr/bin/env python3
"""
test_motor.py — 电机状态接收测试
监听串口，解析 Teensy4.1 上报的 StatusFrame，
将每帧数据打印到终端，并每秒统计一次上报频率。

用法:
    python test_motor.py              # 自动选第一个可用串口
    python test_motor.py COM3
    python test_motor.py COM3 115200  # 自定义波特率
"""

import signal
import struct
import sys
import time

import serial
import serial.tools.list_ports

# ── 协议常量（与 motor_controller.py 保持一致）────────────────────────────────
HEAD_STATUS = 0xCC
FRAME_TAIL  = 0x55
STATUS_FMT  = '<BHBBffffffIBB'
STATUS_SIZE = struct.calcsize(STATUS_FMT)   # 35 bytes
BAUD_DEFAULT = 9600


def calc_crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def parse_frame(data: bytes):
    """解析一帧，返回字段 dict；校验失败返回 None。"""
    if data[0] != HEAD_STATUS or data[-1] != FRAME_TAIL:
        return None
    if calc_crc8(data[1:STATUS_SIZE - 2]) != data[-2]:
        return None
    try:
        _, seq, state, err, lp, lv, lt, rp, rv, rt, ts, crc, tail = \
            struct.unpack(STATUS_FMT, data)
    except struct.error:
        return None
    return dict(seq=seq, state=state, err=err,
                lpos=lp, lvel=lv, ltq=lt,
                rpos=rp, rvel=rv, rtq=rt,
                ts=ts)


def main():
    # ── 串口选择 ──────────────────────────────────────────────────────────────
    port = None
    baud = BAUD_DEFAULT

    if len(sys.argv) >= 2:
        port = sys.argv[1]
    if len(sys.argv) >= 3:
        baud = int(sys.argv[2])

    if port is None:
        # Teensy 4.1 固定 VID/PID：0x16C0 / 0x0483
        TEENSY_VID, TEENSY_PID = 0x16C0, 0x0483
        for p in serial.tools.list_ports.comports():
            if p.vid == TEENSY_VID and p.pid == TEENSY_PID:
                port = p.device
                print(f"[自动识别] Teensy 串口: {port}  ({p.hwid})")
                break

        if port is None:
            # 回退：列出所有串口供参考
            all_ports = serial.tools.list_ports.comports()
            if not all_ports:
                print("[错误] 未找到任何串口，请检查 USB 连接")
            else:
                print("[警告] 未找到 Teensy (VID=16C0 PID=0483)，当前可用串口：")
                for p in all_ports:
                    vid = hex(p.vid) if p.vid else "N/A"
                    pid = hex(p.pid) if p.pid else "N/A"
                    print(f"  {p.device}  VID={vid}  PID={pid}  {p.description}")
            sys.exit(1)

    print(f"打开串口 {port}  波特率 {baud}  帧长 {STATUS_SIZE} bytes")
    print(f"{'seq':>6}  {'state':>5}  {'err':>4}  "
          f"{'lpos':>8}  {'lvel':>8}  {'ltq':>8}  "
          f"{'rpos':>8}  {'rvel':>8}  {'rtq':>8}  "
          f"{'ts(ms)':>10}")
    print("-" * 90)

    try:
        ser = serial.Serial(port, baud, timeout=0.05)
    except serial.SerialException as e:
        print(f"[错误] 无法打开串口: {e}")
        sys.exit(1)

    # ── Ctrl+C 优雅退出 ───────────────────────────────────────────────────────
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    buf          = b''
    frame_count  = 0   # 当前统计周期内收到的帧数
    total_count  = 0   # 累计帧数
    fps          = 0
    fps_last     = time.time()
    crc_errors   = 0

    try:
        while running:
            try:
                chunk = ser.read(128)
            except serial.SerialException as e:
                print(f"\n[串口错误] {e}")
                break

            if not chunk:
                continue

            buf += chunk

            while len(buf) >= STATUS_SIZE:
                idx = buf.find(bytes([HEAD_STATUS]))
                if idx < 0:
                    buf = b''
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < STATUS_SIZE:
                    break

                candidate = buf[:STATUS_SIZE]
                result = parse_frame(candidate)
                buf = buf[STATUS_SIZE:]

                if result is None:
                    crc_errors += 1
                    continue

                frame_count += 1
                total_count += 1

                # ── 打印每帧 ──────────────────────────────────────────────
                print(
                    f"{result['seq']:>6}  "
                    f"{result['state']:>5}  "
                    f"{result['err']:>4}  "
                    f"{result['lpos']:>+8.3f}  "
                    f"{result['lvel']:>+8.3f}  "
                    f"{result['ltq']:>+8.3f}  "
                    f"{result['rpos']:>+8.3f}  "
                    f"{result['rvel']:>+8.3f}  "
                    f"{result['rtq']:>+8.3f}  "
                    f"{result['ts']:>10}",
                    flush=True
                )

            # ── 每秒统计频率 ──────────────────────────────────────────────
            now = time.time()
            if now - fps_last >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_last = now
                print(f"\n>>> 上报频率: {fps} Hz  |  累计帧数: {total_count}  |  CRC错误: {crc_errors}\n",
                      flush=True)

    finally:
        ser.close()
        print(f"\n串口已关闭  累计接收 {total_count} 帧  CRC错误 {crc_errors} 次")


if __name__ == "__main__":
    main()
