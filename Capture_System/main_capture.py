"""
主采集入口 - 同步启动 IMU 和超声两个采集进程

输出目录结构：
    data/
    └── 20260414_153022/           ← 每次运行自动创建（按时间戳命名）
        ├── imu_20260414_153022.csv
        └── ultrasound_20260414_153022.csv

运行：
    # 自动搜索超声设备 IP，采集 IMU + 超声
    python main_capture.py

    # 手动指定超声设备 IP
    python main_capture.py --device-ip 192.168.137.222

    # 指定采集时长 60 秒
    python main_capture.py --device-ip 192.168.137.222 --duration 60

    # 只采集超声（跳过 IMU）
    python main_capture.py --no-imu

    # 只采集 IMU（跳过超声）
    python main_capture.py --no-ultrasound

    # 指定超声通道
    python main_capture.py --channels 1,2
"""

import argparse
import subprocess
import sys
import time
import signal
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="IMU + 超声同步采集系统")
    parser.add_argument("--device-ip",    type=str,   default=None,
                        help="超声设备 IP（不填则自动搜索）")
    parser.add_argument("--channels",     type=str,   default="1,2,3,4",
                        help="超声通道，逗号分隔（默认 '1,2,3,4'）")
    parser.add_argument("--duration",     type=float, default=0,
                        help="采集时长（秒），0 表示持续到 Ctrl+C")
    parser.add_argument("--output-dir",   type=str,   default="./data",
                        help="数据根目录（默认 ./data）")
    parser.add_argument("--no-imu",       action="store_true",
                        help="跳过 IMU 采集")
    parser.add_argument("--no-ultrasound", action="store_true",
                        help="跳过超声采集")
    return parser.parse_args()


def stream_output(proc, prefix: str):
    """将子进程的 stdout 实时转发到当前终端，加上前缀"""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"{prefix} {text}", flush=True)


def main():
    args = parse_args()

    if args.no_imu and args.no_ultrasound:
        print("[错误] --no-imu 和 --no-ultrasound 不能同时使用")
        sys.exit(1)

    # ── 创建本次会话输出目录 ─────────────────────────────────────────────
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / session_tag
    session_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  IMU + 超声同步采集系统")
    print("=" * 60)
    print(f"  会话标签  : {session_tag}")
    print(f"  输出目录  : {session_dir.resolve()}")
    print(f"  采集模式  : {'IMU' if not args.no_imu else ''}{'+'if not args.no_imu and not args.no_ultrasound else ''}{'超声' if not args.no_ultrasound else ''}")
    print(f"  采集时长  : {'持续' if args.duration == 0 else f'{args.duration:.0f} 秒'}")
    if not args.no_ultrasound:
        print(f"  超声 IP   : {args.device_ip or '自动搜索'}")
        print(f"  超声通道  : {args.channels}")
    print("=" * 60)

    python = sys.executable   # 使用当前 Python 解释器，确保环境一致
    script_dir = Path(__file__).parent

    processes = {}    # {"imu": Popen, "ultrasound": Popen}
    threads   = {}    # 日志转发线程

    # ── 启动 IMU 子进程 ──────────────────────────────────────────────────
    if not args.no_imu:
        imu_cmd = [
            python, str(script_dir / "capture_imu.py"),
            "--output-dir",  str(session_dir),
            "--session-tag", session_tag,
        ]
        if args.duration > 0:
            imu_cmd += ["--duration", str(args.duration)]

        print(f"\n[启动] IMU 进程: {' '.join(imu_cmd)}")
        imu_proc = subprocess.Popen(
            imu_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir),
        )
        processes["imu"] = imu_proc

        import threading
        t = threading.Thread(
            target=stream_output, args=(imu_proc, "[IMU]  "), daemon=True
        )
        t.start()
        threads["imu"] = t

    # ── 启动超声子进程 ────────────────────────────────────────────────────
    if not args.no_ultrasound:
        ult_cmd = [
            python, str(script_dir / "capture_ultrasound.py"),
            "--output-dir",  str(session_dir),
            "--session-tag", session_tag,
            "--channels",    args.channels,
        ]
        if args.device_ip:
            ult_cmd += ["--device-ip", args.device_ip]
        if args.duration > 0:
            ult_cmd += ["--duration", str(args.duration)]

        print(f"\n[启动] 超声进程: {' '.join(ult_cmd)}")
        ult_proc = subprocess.Popen(
            ult_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir),
        )
        processes["ultrasound"] = ult_proc

        import threading
        t = threading.Thread(
            target=stream_output, args=(ult_proc, "[超声] "), daemon=True
        )
        t.start()
        threads["ultrasound"] = t

    print(f"\n两个进程已启动，按 Ctrl+C 停止所有采集...\n")
    print("-" * 60)

    # ── 等待 / 监控 ──────────────────────────────────────────────────────
    def stop_all(sig=None, frame=None):
        print("\n\n[停止] 正在终止所有采集进程...")
        for name, proc in processes.items():
            if proc.poll() is None:   # 进程仍在运行
                proc.send_signal(signal.SIGINT)   # 发送 Ctrl+C（优雅退出）
                print(f"  已发送 SIGINT → {name} 进程 (PID {proc.pid})")

    signal.signal(signal.SIGINT, stop_all)

    try:
        if args.duration > 0:
            # 定时采集：等待时长到期，再给子进程一点收尾时间
            time.sleep(args.duration + 3)
            stop_all()
        else:
            # 持续采集：监控子进程，任意一个退出则停止全部
            while True:
                for name, proc in list(processes.items()):
                    ret = proc.poll()
                    if ret is not None:
                        print(f"\n[警告] {name} 进程已退出（返回码 {ret}），停止其他进程")
                        stop_all()
                        break
                else:
                    time.sleep(0.5)
                    continue
                break

    except KeyboardInterrupt:
        stop_all()

    # ── 等待所有子进程完全退出 ────────────────────────────────────────────
    print("\n等待子进程退出...")
    for name, proc in processes.items():
        try:
            proc.wait(timeout=10)
            print(f"  {name} 进程已退出（返回码 {proc.returncode}）")
        except subprocess.TimeoutExpired:
            print(f"  {name} 进程超时未退出，强制终止")
            proc.kill()

    # ── 汇总 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  采集完成")
    print(f"  数据保存在: {session_dir.resolve()}")
    for f in sorted(session_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
