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
import json
import os
import signal
import socket
import subprocess as _subp
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
#  端口占用检查与等待释放
# ─────────────────────────────────────────────

def _is_port_bound(port: int) -> bool:
    """检测 UDP 端口是否已被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def _wait_port_free(port: int, timeout: float = 8.0):
    """
    检测 UDP 端口是否被占用，若占用则用 netstat + taskkill 强制释放（Windows），
    然后轮询等待端口空闲，超时后打印警告继续。
    """
    if not _is_port_bound(port):
        return

    print(f"[超声] 端口 {port} 被占用，尝试释放...", flush=True)

    if sys.platform == "win32":
        import subprocess as _sp
        try:
            # netstat -ano 找出占用该 UDP 端口的 PID
            result = _sp.run(
                f'netstat -ano | findstr ":{port} "',
                shell=True, capture_output=True, text=True
            )
            pids = set()
            for line in result.stdout.splitlines():
                parts = line.split()
                # 格式: 协议 本地地址 外部地址 [状态] PID
                # UDP 行没有状态列，最后一列是 PID
                if f":{port}" in (parts[1] if len(parts) > 1 else ""):
                    pids.add(parts[-1])
            for pid in pids:
                print(f"[超声] taskkill /F /PID {pid}", flush=True)
                _sp.run(f"taskkill /F /PID {pid}", shell=True,
                        capture_output=True)
        except Exception as e:
            print(f"[超声] 释放端口时出错: {e}", flush=True)

    # 轮询等待端口释放
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_port_bound(port):
            print(f"[超声] 端口 {port} 已释放", flush=True)
            return
        time.sleep(0.5)

    print(f"[超声] 警告：端口 {port} 等待超时，继续尝试...", flush=True)


# ─────────────────────────────────────────────
#  mDNS 设备自动搜索
# ─────────────────────────────────────────────

def search_device(timeout: float = 10.0):
    if not ZEROCONF_AVAILABLE:
        return None

    class _Listener(ServiceListener):
        def __init__(self):
            self.found_ips = []
            self._event = threading.Event()

        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name)
            if info:
                ips = info.parsed_addresses()
                self.found_ips.extend(ips)
                self._event.set()

        def update_service(self, zc, type_, name): pass
        def remove_service(self, zc, type_, name): pass

    zeroconf = Zeroconf()
    listener = _Listener()
    browser  = ServiceBrowser(zeroconf, "_http._udp.local.", listener)
    found    = listener._event.wait(timeout=timeout)
    browser.cancel()
    zeroconf.close()

    if found and listener.found_ips:
        ip = listener.found_ips[0]
        return ip
    return None


# ─────────────────────────────────────────────
#  全局状态
# ─────────────────────────────────────────────
_total_packets   = 0
_curr_pack_num   = 0
_csv_writer      = None
_csv_lock        = threading.Lock()

# 预览 socket（由 main() 注入）
_preview_sock    = None          # UDP socket
_preview_addr    = None          # (host, port)
_preview_interval = 10           # 每 N 帧发一帧
_ch_frame_count  = {}            # {ch: 计数器}


# ─────────────────────────────────────────────
#  C# SDK 回调
# ─────────────────────────────────────────────

def _on_notification(packet_type, message):
    pass


def _on_rel_data(is_ult, pack_number):
    global _curr_pack_num
    if is_ult:
        _curr_pack_num = pack_number


def _on_ultrasound_data(ultrasonic_data_by_channel):
    global _total_packets, _csv_writer, _ch_frame_count
    recv_time = time.time()

    for ch, waveforms in ultrasonic_data_by_channel.items():
        for wf in waveforms:
            data = list(wf)
            _total_packets += 1

            # ── 写 CSV ───────────────────────────────────────────────────
            if _csv_writer is not None:
                with _csv_lock:
                    _csv_writer.writerow(
                        [f"{recv_time:.6f}", ch, _curr_pack_num] + data
                    )

            # ── 预览抽帧：每通道独立计数，每 _preview_interval 帧发一帧 ──
            if _preview_sock is not None and _preview_addr is not None:
                cnt = _ch_frame_count.get(ch, 0) + 1
                _ch_frame_count[ch] = cnt
                if cnt % _preview_interval == 0:
                    try:
                        msg = json.dumps({"ch": ch, "data": data}).encode()
                        _preview_sock.sendto(msg, _preview_addr)
                    except Exception:
                        pass


# ─────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Elonxi 超声采集进程")
    parser.add_argument("--device-ip",       type=str,   default=None,
                        help="设备 IP（不填则自动搜索）")
    parser.add_argument("--port",            type=int,   default=1430,
                        help="通信端口（默认 1430）")
    parser.add_argument("--channels",        type=str,   default="1,2,3,4",
                        help="超声通道，逗号分隔（默认 '1,2,3,4'）")
    parser.add_argument("--duration",        type=float, default=0,
                        help="采集时长（秒），0 表示持续到 Ctrl+C")
    parser.add_argument("--output-dir",      type=str,   default="./data",
                        help="CSV 输出目录（默认 ./data）")
    parser.add_argument("--session-tag",     type=str,   default=None,
                        help="文件名时间标签，不填则自动生成")
    parser.add_argument("--preview-port",    type=int,   default=0,
                        help="预览 UDP 端口，0 表示不发送预览（默认 0）")
    parser.add_argument("--preview-interval",type=int,   default=10,
                        help="每隔多少帧发一帧预览（默认 10）")
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
            sys.exit(1)

    # ── 检查 SDK 通信端口是否已被占用，有则等待其释放 ──────────────────────
    _wait_port_free(args.port, timeout=8.0)

    global _csv_writer, _preview_sock, _preview_addr, _preview_interval
    csv_file   = None
    newsletter = None
    _running   = True   # 由信号处理器置 False

    # ── 初始化预览 UDP socket ────────────────────────────────────────────
    if args.preview_port > 0:
        _preview_interval = args.preview_interval
        _preview_addr     = ("127.0.0.1", args.preview_port)
        _preview_sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[超声] 预览 UDP → 127.0.0.1:{args.preview_port}  间隔 {_preview_interval} 帧")

    def _handle_stop(sig, frame):
        nonlocal _running
        _running = False

    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGBREAK"):              # Windows CTRL_BREAK_EVENT
        signal.signal(signal.SIGBREAK, _handle_stop)

    try:
        # ── 打开 CSV ─────────────────────────────────────────────────────────
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

        if args.duration > 0:
            end_time = time.time() + args.duration
            while _running and time.time() < end_time:
                time.sleep(0.1)
        else:
            while _running:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止采集")

    finally:
        print("\n[超声] 正在释放资源...")
        # 1. 先停止采集、再断开设备（newsletter 内部持有 UDP socket，这里释放它）
        if newsletter is not None:
            try:
                newsletter.collectionSwitch(False)
                time.sleep(0.3)
                newsletter.deviceSwitch(False)
                time.sleep(0.5)   # 给 SDK 内部 socket 多一点关闭时间
                print("[超声] 设备已断开")
            except Exception as e:
                print(f"[超声] 关闭设备时出错: {e}", file=sys.stderr)
            finally:
                # 强制置 None，让 GC / .NET finalizer 释放内部 socket
                newsletter = None
        # 2. 注销事件回调，防止残留回调写已关闭的 CSV
        try:
            GlobalEvents.NotificationReceived     -= _on_notification
            GlobalEvents.RealRealUltrDataReceived -= _on_ultrasound_data
            GlobalEvents.RealRealRelDataReceived  -= _on_rel_data
        except Exception:
            pass
        # 3. 关闭 CSV（先置 None 让回调不再写入）
        _csv_writer = None
        if csv_file is not None:
            try:
                csv_file.flush()
                csv_file.close()
            except Exception:
                pass
        # 4. 关闭预览 socket
        if _preview_sock is not None:
            try:
                _preview_sock.close()
            except Exception:
                pass
            _preview_sock = None
        print(f"[超声] 采集结束，共 {_total_packets} 包  →  {csv_path}")


if __name__ == "__main__":
    main()
