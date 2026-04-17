"""
IMU + 超声同步采集系统 — 可视化界面（合并版）
直接启动 UI，点击"开始采集"填写受试者信息后拉起
capture_imu.py / capture_ultrasound.py，
点击"停止采集"或关闭窗口时优雅终止两个子进程并释放资源。

运行:
    python UI_main_capture.py
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from PySide6.QtCore import QThread, Signal, Qt
    from PySide6.QtGui import QFont, QTextCursor
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
        QPushButton, QLabel, QTextEdit, QSpinBox, QGroupBox, QDoubleSpinBox,
        QDialog, QDialogButtonBox, QLineEdit, QComboBox, QMessageBox,
    )
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal as Signal, Qt
    from PyQt5.QtGui import QFont, QTextCursor
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
        QPushButton, QLabel, QTextEdit, QSpinBox, QGroupBox, QDoubleSpinBox,
        QDialog, QDialogButtonBox, QLineEdit, QComboBox, QMessageBox,
    )

import pyqtgraph as pg

SCRIPT_DIR   = Path(__file__).parent
PREVIEW_PORT = 19876    # UI 监听的 UDP 端口，capture_ultrasound.py 向此发送预览帧
UDP_BUF      = 65536

# ── 下拉框选项（按需修改）────────────────────────────────────────────────────
CAPTURE_CONDITIONS = [  "1_1.平地行走0.8m/s",
                        "1_2.平地行走1.25m/s", 
                        "1_3.平地行走1.6m/s", 
                        "2.上坡5°,1m/s", 
                        "3.下坡5°,1m/s", 
                        "4.上楼", 
                        "5.下楼", 
                        "6_1.站立", 
                        "6_2.站立2行走",
                        "6_3.行走2站立",
                      ]
WEARING_CONDITIONS = ["a.不穿", "b.零力矩", "c.端到端助力", "d.样条助力"]


# ─────────────────────────────────────────────────────────────────────────────
#  工具：解码子进程输出行
# ─────────────────────────────────────────────────────────────────────────────

def _decode_line(raw: bytes) -> str:
    """优先 utf-8，回退 gbk，再回退 latin-1；过滤纯十六进制 SDK 调试行。"""
    text = ""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc).rstrip()
            break
        except UnicodeDecodeError:
            continue
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) > 8 and all(c in "0123456789ABCDEFabcdef"
                                  for c in stripped.replace(" ", "")):
        return ""
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  受试者信息输入对话框
# ─────────────────────────────────────────────────────────────────────────────

class SubjectInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("受试者信息 — 开始采集前请填写完整")
        self.setMinimumWidth(380)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(8)

        # 受试者编号
        self._edit_subject = QLineEdit()
        self._edit_subject.setPlaceholderText("例如：S001")
        form.addRow("受试者编号 *", self._edit_subject)

        # 身高（cm）
        self._spin_height = QDoubleSpinBox()
        self._spin_height.setRange(50.0, 250.0)
        self._spin_height.setValue(170.0)
        self._spin_height.setDecimals(1)
        self._spin_height.setSuffix("  cm")
        form.addRow("身高 *", self._spin_height)

        # 体重（kg）
        self._spin_weight = QDoubleSpinBox()
        self._spin_weight.setRange(10.0, 300.0)
        self._spin_weight.setValue(65.0)
        self._spin_weight.setDecimals(1)
        self._spin_weight.setSuffix("  kg")
        form.addRow("体重 *", self._spin_weight)

        # 年龄
        self._spin_age = QSpinBox()
        self._spin_age.setRange(1, 120)
        self._spin_age.setValue(25)
        self._spin_age.setSuffix("  岁")
        form.addRow("年龄 *", self._spin_age)

        # 采集条件（下拉）
        self._combo_capture = QComboBox()
        self._combo_capture.addItems(CAPTURE_CONDITIONS)
        form.addRow("采集条件 *", self._combo_capture)

        # 穿戴条件（下拉）
        self._combo_wearing = QComboBox()
        self._combo_wearing.addItems(WEARING_CONDITIONS)
        form.addRow("穿戴条件 *", self._combo_wearing)

        # Session（仅数字）
        self._spin_session = QSpinBox()
        self._spin_session.setRange(1, 9999)
        self._spin_session.setValue(1)
        self._spin_session.setPrefix("第 ")
        self._spin_session.setSuffix(" 次")
        form.addRow("Session *", self._spin_session)

        layout.addLayout(form)

        # 确定 / 取消按钮
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btns.button(QDialogButtonBox.Ok).setText("确定，开始采集")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        if not self._edit_subject.text().strip():
            QMessageBox.warning(self, "提示", "请填写受试者编号")
            return
        self.accept()

    # ── 获取填写结果 ──────────────────────────────────────────────────────────
    def get_info(self) -> dict:
        return {
            "subject_id":        self._edit_subject.text().strip(),
            "height_cm":         self._spin_height.value(),
            "weight_kg":         self._spin_weight.value(),
            "age":               self._spin_age.value(),
            "capture_condition": self._combo_capture.currentText(),
            "wearing_condition": self._combo_wearing.currentText(),
            "session":           self._spin_session.value(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  后台线程：读取单个子进程的 stdout，通过 Signal 发到主线程
# ─────────────────────────────────────────────────────────────────────────────

class ProcReaderThread(QThread):
    line_ready   = Signal(str)
    finished_sig = Signal(str, int)  # (name, retcode)

    def __init__(self, proc: subprocess.Popen, name: str, prefix: str):
        super().__init__()
        self._proc   = proc
        self._name   = name
        self._prefix = prefix

    def run(self):
        for raw in iter(self._proc.stdout.readline, b""):
            text = _decode_line(raw)
            if text:
                self.line_ready.emit(f"{self._prefix} {text}")
        ret = self._proc.wait()
        self.finished_sig.emit(self._name, ret)


# ─────────────────────────────────────────────────────────────────────────────
#  后台线程：UDP 接收超声预览帧
# ─────────────────────────────────────────────────────────────────────────────

class UdpReceiverThread(QThread):
    wave_ready = Signal(int, list)   # (channel, [1000点])

    def __init__(self, port: int):
        super().__init__()
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
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


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._processes:   dict[str, subprocess.Popen]    = {}
        self._readers:     dict[str, ProcReaderThread]    = {}
        self._udp:         UdpReceiverThread | None       = None
        self._session_dir: Path | None                    = None
        self._subject_info: dict                          = {}
        self._stopped = False

        self._curves: dict[int, pg.PlotDataItem] = {}
        self._build_ui()

    # ── 界面构建 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("IMU + 超声同步采集系统")
        self.resize(980, 760)

        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # 标题
        title = QLabel("IMU + 超声同步采集系统")
        title.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(14); f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        # 受试者信息展示行
        self._lbl_subject = QLabel("受试者：未填写")
        self._lbl_subject.setAlignment(Qt.AlignCenter)
        self._lbl_subject.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._lbl_subject)

        # IMU 设备数展示行
        self._lbl_imu_devices = QLabel("IMU 设备：未连接")
        self._lbl_imu_devices.setAlignment(Qt.AlignCenter)
        self._lbl_imu_devices.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._lbl_imu_devices)

        # 状态
        self._status = QLabel("状态：未采集")
        self._status.setAlignment(Qt.AlignCenter)
        root.addWidget(self._status)

        # 控制行
        ctrl = QHBoxLayout()
        self._btn_start = QPushButton("开始采集")
        self._btn_stop  = QPushButton("停止采集")
        self._btn_stop.setEnabled(False)
        for b in (self._btn_start, self._btn_stop):
            b.setFixedHeight(34)
        self._btn_start.clicked.connect(self._on_start_clicked)
        self._btn_stop.clicked.connect(self._on_stop_clicked)

        self._spin_interval = QSpinBox()
        self._spin_interval.setRange(1, 200)
        self._spin_interval.setValue(10)
        self._spin_interval.setFixedWidth(70)

        # Y 轴范围
        self._spin_ymin = QDoubleSpinBox()
        self._spin_ymin.setRange(-1e6, 1e6)
        self._spin_ymin.setValue(0)
        self._spin_ymin.setFixedWidth(80)
        self._spin_ymin.setDecimals(0)

        self._spin_ymax = QDoubleSpinBox()
        self._spin_ymax.setRange(-1e6, 1e6)
        self._spin_ymax.setValue(300)
        self._spin_ymax.setFixedWidth(80)
        self._spin_ymax.setDecimals(0)

        btn_apply_y = QPushButton("应用")
        btn_apply_y.setFixedWidth(50)
        btn_apply_y.setFixedHeight(28)
        btn_apply_y.clicked.connect(self._apply_yrange)

        # X 轴范围
        self._spin_xmin = QSpinBox()
        self._spin_xmin.setRange(0, 999)
        self._spin_xmin.setValue(0)
        self._spin_xmin.setFixedWidth(70)

        self._spin_xmax = QSpinBox()
        self._spin_xmax.setRange(0, 999)
        self._spin_xmax.setValue(999)
        self._spin_xmax.setFixedWidth(70)

        btn_apply_x = QPushButton("应用")
        btn_apply_x.setFixedWidth(50)
        btn_apply_x.setFixedHeight(28)
        btn_apply_x.clicked.connect(self._apply_xrange)

        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_stop)
        ctrl.addStretch()
        ctrl.addWidget(QLabel("X轴:"))
        ctrl.addWidget(self._spin_xmin)
        ctrl.addWidget(QLabel("~"))
        ctrl.addWidget(self._spin_xmax)
        ctrl.addWidget(btn_apply_x)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Y轴:"))
        ctrl.addWidget(self._spin_ymin)
        ctrl.addWidget(QLabel("~"))
        ctrl.addWidget(self._spin_ymax)
        ctrl.addWidget(btn_apply_y)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("预览间隔（帧）:"))
        ctrl.addWidget(self._spin_interval)
        root.addLayout(ctrl)

        # 波形区（4 通道 2×2）
        wave_group = QGroupBox("超声波形预览")
        wl = QVBoxLayout(wave_group); wl.setSpacing(4)
        row1 = QHBoxLayout(); row1.setSpacing(4)
        row2 = QHBoxLayout(); row2.setSpacing(4)

        pg.setConfigOptions(antialias=False, background="k", foreground="w")
        self._plot_widgets: dict[int, pg.PlotWidget] = {}
        for ch in (1, 2, 3, 4):
            pw = pg.PlotWidget(title=f"通道 {ch}")
            pw.setLabel("left", "幅值")
            pw.setLabel("bottom", "采样点")
            pw.setXRange(0, 999, padding=0)
            pw.setYRange(0, 300, padding=0)
            pw.enableAutoRange(axis="y", enable=False)
            pw.showGrid(x=False, y=True, alpha=0.3)
            curve = pw.plot(pen=pg.mkPen(color=(80, 200, 120), width=1))
            self._curves[ch] = curve
            self._plot_widgets[ch] = pw
            (row1 if ch <= 2 else row2).addWidget(pw)
        wl.addLayout(row1)
        wl.addLayout(row2)
        root.addWidget(wave_group, stretch=3)

        # 日志区
        log_group = QGroupBox("运行日志")
        ll = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setMaximumHeight(160)
        ll.addWidget(self._log)
        root.addWidget(log_group, stretch=1)

    # ── X/Y 轴范围应用 ────────────────────────────────────────────────────────
    def _apply_xrange(self):
        xmin, xmax = self._spin_xmin.value(), self._spin_xmax.value()
        if xmin >= xmax:
            return
        for pw in self._plot_widgets.values():
            pw.setXRange(xmin, xmax, padding=0)

    def _apply_yrange(self):
        ymin, ymax = self._spin_ymin.value(), self._spin_ymax.value()
        if ymin >= ymax:
            return
        for pw in self._plot_widgets.values():
            pw.setYRange(ymin, ymax, padding=0)

    # ── 点击"开始采集"：先弹对话框 ────────────────────────────────────────────
    def _on_start_clicked(self):
        dlg = SubjectInfoDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return

        self._subject_info = dlg.get_info()
        self._start(self._subject_info)

    # ── 正式启动采集 ──────────────────────────────────────────────────────────
    def _start(self, info: dict):
        self._log.clear()
        for c in self._curves.values():
            c.setData([])
        self._stopped = False

        # 构造文件夹名：[受试者编号]_[穿戴条件]_[采集条件]_[时间戳]
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_tag = timestamp
        folder_name = (
            f"{info['subject_id']}"
            f"_{info['wearing_condition']}"
            f"_{info['capture_condition']}"
            f"_{timestamp}"
        )
        self._session_dir = SCRIPT_DIR / "data" / folder_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # 将受试者信息保存为 JSON
        meta_path = self._session_dir / "subject_info.json"
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(info, fp, ensure_ascii=False, indent=2)

        # 更新标题信息栏
        self._lbl_subject.setText(
            f"受试者: {info['subject_id']}  |  "
            f"身高: {info['height_cm']} cm  体重: {info['weight_kg']} kg  年龄: {info['age']} 岁  |  "
            f"采集条件: {info['capture_condition']}  穿戴条件: {info['wearing_condition']}  "
            f"Session: {info['session']}"
        )
        self._lbl_subject.setStyleSheet("color: #90EE90; font-size: 11px;")
        self._lbl_imu_devices.setText("IMU 设备：连接中...")
        self._lbl_imu_devices.setStyleSheet("color: gray; font-size: 11px;")

        python   = sys.executable
        interval = self._spin_interval.value()

        # ── 先启动 UDP 接收线程 ────────────────────────────────────────────
        self._udp = UdpReceiverThread(PREVIEW_PORT)
        self._udp.wave_ready.connect(self._on_wave)
        self._udp.start()

        # ── 启动 IMU 子进程 ───────────────────────────────────────────────
        imu_cmd = [
            python, str(SCRIPT_DIR / "capture_imu.py"),
            "--output-dir",  str(self._session_dir),
            "--session-tag", session_tag,
        ]
        imu_proc = subprocess.Popen(
            imu_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self._processes["imu"] = imu_proc
        imu_reader = ProcReaderThread(imu_proc, "imu", "[IMU] ")
        imu_reader.line_ready.connect(self._append_log)
        imu_reader.line_ready.connect(self._on_imu_log)
        imu_reader.finished_sig.connect(self._on_proc_finished)
        imu_reader.start()
        self._readers["imu"] = imu_reader

        # ── 启动超声子进程 ────────────────────────────────────────────────
        ult_cmd = [
            python, str(SCRIPT_DIR / "capture_ultrasound.py"),
            "--output-dir",       str(self._session_dir),
            "--session-tag",      session_tag,
            "--channels",         "1,2,3,4",
            "--preview-port",     str(PREVIEW_PORT),
            "--preview-interval", str(interval),
        ]
        ult_proc = subprocess.Popen(
            ult_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self._processes["ultrasound"] = ult_proc
        ult_reader = ProcReaderThread(ult_proc, "ultrasound", "[超声]")
        ult_reader.line_ready.connect(self._append_log)
        ult_reader.finished_sig.connect(self._on_proc_finished)
        ult_reader.start()
        self._readers["ultrasound"] = ult_reader

        # ── 启动电机采集子进程 ────────────────────────────────────────────
        motor_cmd = [
            python, str(SCRIPT_DIR / "capture_motor.py"),
            "--output-dir",  str(self._session_dir),
            "--session-tag", session_tag,
        ]
        motor_proc = subprocess.Popen(
            motor_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        self._processes["motor"] = motor_proc
        motor_reader = ProcReaderThread(motor_proc, "motor", "[电机]")
        motor_reader.line_ready.connect(self._append_log)
        motor_reader.finished_sig.connect(self._on_proc_finished)
        motor_reader.start()
        self._readers["motor"] = motor_reader

        self._append_log(f"会话目录: {self._session_dir}")
        self._append_log(f"受试者信息已保存 → subject_info.json")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._spin_interval.setEnabled(False)
        self._status.setText("状态：采集中...")

    # ── IMU 日志解析：提取设备连接数 ─────────────────────────────────────────
    def _on_imu_log(self, text: str):
        # capture_imu.py 打印：共连接 X 个 MTw 设备
        if "共连接" in text and "MTw" in text:
            try:
                n = int(text.split("共连接")[1].split("个")[0].strip())
                self._lbl_imu_devices.setText(f"IMU 设备：已连接 {n} 个 MTw")
                self._lbl_imu_devices.setStyleSheet("color: #90EE90; font-size: 11px;")
            except (ValueError, IndexError):
                pass

    # ── 停止采集（按钮触发） ──────────────────────────────────────────────────
    def _on_stop_clicked(self):
        self._btn_stop.setEnabled(False)
        self._status.setText("状态：正在停止...")
        self._stop_all()

    def _stop_all(self):
        if self._stopped:
            return
        self._stopped = True
        for name, proc in self._processes.items():
            if proc.poll() is None:
                try:
                    if sys.platform == "win32":
                        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
                    else:
                        proc.send_signal(signal.SIGINT)
                    self._append_log(f"[停止] 已发送终止信号 → {name} (PID {proc.pid})")
                except Exception as e:
                    self._append_log(f"[停止] 发送信号失败 {name}: {e}")

    # ── 子进程退出回调 ────────────────────────────────────────────────────────
    def _on_proc_finished(self, name: str, retcode: int):
        self._append_log(f"[{name}] 进程已退出（返回码 {retcode}）")
        self._processes.pop(name, None)
        self._readers.pop(name, None)

        # 电机进程找不到设备时正常退出（retcode=1），不影响其他采集
        if name == "motor" and retcode == 1:
            self._append_log("[电机] 未找到 Teensy，电机数据不采集，其他采集继续")
        elif not self._stopped and retcode not in (0, -2, -15):
            self._append_log(f"[警告] {name} 意外退出，停止所有采集")
            self._stop_all()

        if len(self._processes) == 0:
            self._on_all_finished()

    def _on_all_finished(self):
        if self._udp:
            self._udp.stop()
            self._udp.wait(2000)
            self._udp = None

        if self._session_dir and self._session_dir.exists():
            self._append_log("=" * 50)
            self._append_log(f"数据保存在: {self._session_dir}")
            for f in sorted(self._session_dir.iterdir()):
                self._append_log(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
            self._append_log("=" * 50)

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._spin_interval.setEnabled(True)
        self._status.setText("状态：已停止")
        self._lbl_imu_devices.setText("IMU 设备：未连接")
        self._lbl_imu_devices.setStyleSheet("color: gray; font-size: 11px;")

    # ── 波形更新 ──────────────────────────────────────────────────────────────
    def _on_wave(self, ch: int, data: list):
        curve = self._curves.get(ch)
        if curve is not None:
            curve.setData(data)

    # ── 日志追加 ──────────────────────────────────────────────────────────────
    def _append_log(self, text: str):
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(text + "\n")
        self._log.moveCursor(QTextCursor.End)

    # ── 关闭窗口 ──────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._stop_all()
        deadline = time.time() + 15
        while self._processes and time.time() < deadline:
            time.sleep(0.2)
        for name, proc in list(self._processes.items()):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if self._udp:
            self._udp.stop()
            self._udp.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
