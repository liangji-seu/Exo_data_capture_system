#!/usr/bin/env python3
"""
capture_motor.py — 电机状态采集进程
由 UI_main_capture.py 通过 subprocess 启动，也可单独运行。

CSV 输出格式（每行一帧）：
    timestamp, seq, state, err, lpos, lvel, ltq, rpos, rvel, rtq, ts

运行：
    python capture_motor.py
    python capture_motor.py --output-dir ./data/20260417_120000 --session-tag 20260417_120000
    python capture_motor.py --duration 60
"""

import argparse
import csv
import signal
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import serial
import serial.tools.list_ports

# ── 协议常量 ──────────────────────────────────────────────────────────────────
HEAD_STATUS  = 0xCC
FRAME_TAIL   = 0x55
STATUS_FMT   = '<BHBBffffffIBB'
STATUS_SIZE  = struct.calcsize(STATUS_FMT)   # 35 bytes
BAUD_DEFAULT = 9600
TEENSY_VID   = 0x16C0
TEENSY_PID   = 0x0483

CSV_HEADER = [
    "timestamp",        # PC 收到该批数据的时间（秒，float）
    "timestamp_hw_ms",  # Teensy 硬件时间戳（毫秒，来自帧内 ts 字段）
    "seq", "state", "err",
    "lpos", "lvel", "ltq",
    "rpos", "rvel", "rtq",
]


def calc_crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def parse_frame(data: bytes):
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


def find_teensy_port():
    for p in serial.tools.list_ports.comports():
        if p.vid == TEENSY_VID and p.pid == TEENSY_PID:
            return p.device
    return None


def main():
    parser = argparse.ArgumentParser(description="电机状态采集进程")
    parser.add_argument("--output-dir",  type=str, default="./data",
                        help="CSV 输出目录（默认 ./data）")
    parser.add_argument("--session-tag", type=str, default=None,
                        help="文件名时间标签，不填则自动生成")
    parser.add_argument("--duration",    type=float, default=0,
                        help="采集时长（秒），0 表示持续到 Ctrl+C（默认 0）")
    parser.add_argument("--port",        type=str, default=None,
                        help="串口号，不填则自动识别 Teensy")
    parser.add_argument("--baud",        type=int, default=BAUD_DEFAULT,
                        help=f"波特率（默认 {BAUD_DEFAULT}）")
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    session_tag = args.session_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"motor_{session_tag}.csv"

    # ── 串口自动识别 ──────────────────────────────────────────────────────────
    port = args.port
    if not port:
        port = find_teensy_port()
        if port:
            print(f"[电机] 自动识别 Teensy 串口: {port}", flush=True)
        else:
            print("[电机] 未找到 Teensy (VID=16C0 PID=0483)，当前可用串口：", flush=True)
            for p in serial.tools.list_ports.comports():
                vid = hex(p.vid) if p.vid else "N/A"
                pid = hex(p.pid) if p.pid else "N/A"
                print(f"  {p.device}  VID={vid}  PID={pid}  {p.description}", flush=True)
            sys.exit(1)

    print(f"[电机] 串口={port}  波特率={args.baud}  输出={csv_path}", flush=True)

    try:
        ser = serial.Serial(port, args.baud, timeout=0.05)
    except serial.SerialException as e:
        print(f"[电机] 无法打开串口: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # ── 信号处理 ──────────────────────────────────────────────────────────────
    _running = True
    def _handle_stop(sig, frame):
        nonlocal _running
        _running = False
    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_stop)

    buf         = b''
    total       = 0
    crc_errors  = 0
    fps_count   = 0
    fps_last    = time.time()
    csv_file    = None

    try:
        csv_file = open(csv_path, "w", newline="", buffering=1)
        writer   = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        print(f"[电机] CSV 已创建: {csv_path}", flush=True)
        print("[电机] 采集中（Ctrl+C 停止）...", flush=True)

        start_time = time.time()
        end_time   = start_time + args.duration if args.duration > 0 else None

        while _running:
            if end_time and time.time() >= end_time:
                print(f"[电机] 已达到采集时长 {args.duration:.0f}s，停止", flush=True)
                break

            try:
                chunk = ser.read(128)
            except serial.SerialException as e:
                print(f"[电机] 串口错误: {e}", flush=True)
                break

            if not chunk:
                continue

            recv_time = time.time()   # 记录本批数据的到达时间
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

                result = parse_frame(buf[:STATUS_SIZE])
                buf = buf[STATUS_SIZE:]

                if result is None:
                    crc_errors += 1
                    continue

                total     += 1
                fps_count += 1

                writer.writerow([
                    f"{recv_time:.6f}",
                    result["ts"],
                    result["seq"], result["state"], result["err"],
                    result["lpos"], result["lvel"], result["ltq"],
                    result["rpos"], result["rvel"], result["rtq"],
                ])

                # 每 200 帧打印一次进度
                if total % 200 == 0:
                    print(f"  [电机] 已写入 {total} 帧", flush=True)

            # 每秒统计频率
            now = time.time()
            if now - fps_last >= 1.0:
                print(f"  [电机] 上报频率: {fps_count} Hz  累计: {total} 帧  CRC错误: {crc_errors}",
                      flush=True)
                fps_count = 0
                fps_last  = now

    except KeyboardInterrupt:
        print("\n[电机] 收到 Ctrl+C，停止采集", flush=True)
    finally:
        if csv_file:
            csv_file.close()
        ser.close()
        print(f"[电机] 采集结束，共 {total} 帧  CRC错误 {crc_errors} 次  →  {csv_path}",
              flush=True)


if __name__ == "__main__":
    main()
