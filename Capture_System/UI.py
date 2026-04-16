"""
IMU + 超声同步采集系统 — 可视化界面
运行: python UI.py
"""

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

try:
    from PySide6.QtCore import QThread, QTimer, Signal, Qt
    from PySide6.QtGui import QFont, QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QSpinBox, QGroupBox,
    )
except ImportError:
    from PyQt5.QtCore import QThread, QTimer, pyqtSignal as Signal, Qt
    from PyQt5.QtGui import QFont, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QTextEdit, QSpinBox, QGroupBox,
    )

import pyqtgraph as pg

SCRIPT        = Path(__file__).parent / "main_capture.py"
PREVIEW_PORT  = 19876          # 固定 UDP 端口，UI 监听，capture_ultrasound.py 发送
UDP_BUF       = 65536          # UDP 接收缓冲


# ── 后台线程：读取子进程 stdout ───────────────────────────────────────────
class ReaderThread(QThread):
    line_ready   = Signal(str)
    finished_sig = Signal(int)

    def __init__(self, proc: subprocess.Popen):
        super().__init__()
        self._proc = proc

    def run(self):
        for raw in iter(self._proc.stdout.readline, b""):
            # 优先 utf-8，回退 gbk，再回退 latin-1
            for enc in ("utf-8", "gbk", "latin-1"):
                try:
                    text = raw.decode(enc).rstrip()
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                continue
            # 过滤 SDK 原始字节行（纯十六进制长串，无实际含义）
            stripped = text.strip()
            if len(stripped) > 8 and all(c in "0123456789ABCDEFabcdef" for c in stripped.replace(" ", "")):
                continue
            self.line_ready.emit(text)
        ret = self._proc.wait()
        self.finished_sig.emit(ret)


# ── 后台线程：UDP 接收预览数据 ───────────────────────────────────────────
class UdpReceiverThread(QThread):
    """在独立线程阻塞接收 UDP，解析后用信号发到主线程"""
    wave_ready = Signal(int, list)   # (channel, [1000点])

    def __init__(self, port: int):
        super().__init__()
        self._port    = port
        self._running = True
        self._sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", self._port))
        self._sock.settimeout(0.5)

    def run(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(UDP_BUF)
                obj = json.loads(data.decode())
                self.wave_ready.emit(int(obj["ch"]), obj["data"])
            except socket.timeout:
                continue
            except Exception:
                continue

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass


# ── 主窗口 ────────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._proc:     subprocess.Popen | None = None
        self._reader:   ReaderThread      | None = None
        self._udp:      UdpReceiverThread | None = None
        self._plots:    dict[int, pg.PlotWidget] = {}   # {ch: PlotWidget}
        self._curves:   dict[int, pg.PlotDataItem] = {} # {ch: curve}
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle("IMU + 超声同步采集系统")
        self.resize(900, 700)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── 标题 ──────────────────────────────────────────────────────────
        title = QLabel("IMU + 超声同步采集系统")
        title.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(14); f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        # ── 状态 ──────────────────────────────────────────────────────────
        self._status = QLabel("状态：未采集")
        self._status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status)

        # ── 控制行 ────────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._btn_start = QPushButton("开始采集")
        self._btn_stop  = QPushButton("停止采集")
        self._btn_stop.setEnabled(False)
        for b in (self._btn_start, self._btn_stop):
            b.setFixedHeight(34)
        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._stop)

        interval_label = QLabel("预览间隔（帧）:")
        self._spin_interval = QSpinBox()
        self._spin_interval.setRange(1, 200)
        self._spin_interval.setValue(10)
        self._spin_interval.setFixedWidth(70)

        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_stop)
        ctrl.addStretch()
        ctrl.addWidget(interval_label)
        ctrl.addWidget(self._spin_interval)
        root.addLayout(ctrl)

        # ── 波形区域：4 通道 2×2 排列 ─────────────────────────────────────
        wave_group = QGroupBox("超声波形预览")
        wave_layout = QVBoxLayout(wave_group)
        wave_layout.setSpacing(4)

        row1 = QHBoxLayout(); row1.setSpacing(4)
        row2 = QHBoxLayout(); row2.setSpacing(4)

        pg.setConfigOptions(antialias=False, background="k", foreground="w")

        for ch in (1, 2, 3, 4):
            pw = pg.PlotWidget(title=f"通道 {ch}")
            pw.setLabel("left", "幅值")
            pw.setLabel("bottom", "采样点")
            pw.setXRange(0, 999, padding=0)
            pw.setYRange(0, 300, padding=0)
            pw.enableAutoRange(axis="y", enable=False)
            pw.showGrid(x=False, y=True, alpha=0.3)
            curve = pw.plot(pen=pg.mkPen(color=(80, 200, 120), width=1))
            self._plots[ch]  = pw
            self._curves[ch] = curve
            if ch in (1, 2):
                row1.addWidget(pw)
            else:
                row2.addWidget(pw)

        wave_layout.addLayout(row1)
        wave_layout.addLayout(row2)
        root.addWidget(wave_group, stretch=3)

        # ── 日志区域 ──────────────────────────────────────────────────────
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setMaximumHeight(160)
        log_layout.addWidget(self._log)
        root.addWidget(log_group, stretch=1)

    # ── 开始采集 ──────────────────────────────────────────────────────────
    def _start(self):
        self._log.clear()
        for curve in self._curves.values():
            curve.setData([])

        # 先启动 UDP 接收线程，再启动子进程（避免丢帧）
        self._udp = UdpReceiverThread(PREVIEW_PORT)
        self._udp.wave_ready.connect(self._on_wave)
        self._udp.start()

        interval = self._spin_interval.value()
        cmd = [
            sys.executable, str(SCRIPT),
            "--preview-port",     str(PREVIEW_PORT),
            "--preview-interval", str(interval),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT.parent),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self._reader = ReaderThread(self._proc)
        self._reader.line_ready.connect(self._append_log)
        self._reader.finished_sig.connect(self._on_finished)
        self._reader.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._spin_interval.setEnabled(False)
        self._status.setText("状态：采集中...")

    # ── 停止采集 ──────────────────────────────────────────────────────────
    def _stop(self):
        self._btn_stop.setEnabled(False)
        self._status.setText("状态：正在停止...")
        if self._proc and self._proc.poll() is None:
            if sys.platform == "win32":
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._proc.send_signal(signal.SIGINT)

    # ── 子进程退出回调 ────────────────────────────────────────────────────
    def _on_finished(self, retcode: int):
        if self._udp:
            self._udp.stop()
            self._udp.wait(2000)
            self._udp = None
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._spin_interval.setEnabled(True)
        self._status.setText(f"状态：已停止（返回码 {retcode}）")
        self._proc = None

    # ── 更新波形 ──────────────────────────────────────────────────────────
    def _on_wave(self, ch: int, data: list):
        curve = self._curves.get(ch)
        if curve is not None:
            curve.setData(data)

    # ── 追加日志 ──────────────────────────────────────────────────────────
    def _append_log(self, text: str):
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(text + "\n")
        self._log.moveCursor(QTextCursor.End)

    # ── 关闭窗口时确保子进程退出 ──────────────────────────────────────────
    def closeEvent(self, event):
        if self._proc and self._proc.poll() is None:
            self._stop()
            if self._reader:
                self._reader.wait(3000)
        if self._udp:
            self._udp.stop()
            self._udp.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
