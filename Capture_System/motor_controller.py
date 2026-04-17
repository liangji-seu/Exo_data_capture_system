#!/usr/bin/env python3
"""
motor_controller.py - tkinter 上位机界面 (ExoCode 协议)
依赖: pip install pyserial
"""

import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import struct
import threading
import time
import atexit

# ==================== 协议常量 ====================
HEAD_CONTROL = 0xAA
HEAD_STATUS  = 0xCC
HEAD_ENABLE  = 0xBB
FRAME_TAIL   = 0x55

# StatusFrame: head(1) seq(2) state(1) err(1) lpos(4f) lvel(4f) ltq(4f) rpos(4f) rvel(4f) rtq(4f) time(4u) crc(1) tail(1) = 35 bytes
STATUS_FMT  = '<BHBBffffffIBB'
STATUS_SIZE = struct.calcsize(STATUS_FMT)   # 35

TORQUE_MAX  = 9.0
TORQUE_STEP = 0.1
SEND_HZ     = 200


# ==================== 协议函数 ====================
def calc_crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_control_frame(seq: int, l_tq: float, r_tq: float, limit: float) -> bytes:
    payload = struct.pack('<H3f', seq, l_tq, r_tq, limit)
    crc = calc_crc8(payload)
    return struct.pack('B', HEAD_CONTROL) + payload + struct.pack('BB', crc, FRAME_TAIL)


def build_enable_frame(cmd: int) -> bytes:
    crc = calc_crc8(bytes([cmd]))
    return struct.pack('BBBB', HEAD_ENABLE, cmd, crc, FRAME_TAIL)


# ==================== 主应用 ====================
class MotorControlApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('ExoSkeleton Motor Control')
        self.root.resizable(False, False)

        self.ser = None
        self.connected = False
        self.enabled = False
        self.seq = 0

        self.left_cmd  = 0.0
        self.right_cmd = 0.0
        self.limit     = 5.0

        self.lpos = self.lvel = self.ltq = 0.0
        self.rpos = self.rvel = self.rtq = 0.0
        self.fps = 0
        self._fps_count = 0
        self._fps_last = time.time()

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._build_ui()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        atexit.register(self._safe_disable)

    # -------------------- UI --------------------
    def _build_ui(self):
        P = dict(padx=6, pady=4)

        # top bar
        top = tk.Frame(self.root)
        top.pack(fill='x', **P)
        tk.Label(top, text='Port:').pack(side='left')
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(top, textvariable=self.port_var, width=10, state='readonly')
        self.port_cb.pack(side='left', padx=4)
        tk.Button(top, text='刷新', command=self._refresh_ports).pack(side='left')
        self.btn_connect = tk.Button(top, text='连接', width=6, command=self._toggle_connect)
        self.btn_connect.pack(side='left', padx=6)
        self.dot = tk.Label(top, text='●', fg='gray', font=('Arial', 14))
        self.dot.pack(side='left')
        self.conn_lbl = tk.Label(top, text='未连接', width=18, anchor='w')
        self.conn_lbl.pack(side='left')

        # motor panels
        panels = tk.Frame(self.root)
        panels.pack(fill='x', padx=6, pady=2)
        self._build_motor_panel(panels, 'LEFT  电机', 'left').pack(side='left', fill='both', expand=True, padx=4)
        self._build_motor_panel(panels, 'RIGHT 电机', 'right').pack(side='left', fill='both', expand=True, padx=4)

        # global settings
        bot = tk.LabelFrame(self.root, text='全局设置')
        bot.pack(fill='x', padx=10, pady=4)
        tk.Label(bot, text='扭矩限制 (Nm):').grid(row=0, column=0, sticky='e', **P)
        self.limit_var = tk.DoubleVar(value=self.limit)
        tk.Scale(bot, from_=0, to=TORQUE_MAX, resolution=0.1, orient='horizontal',
                 variable=self.limit_var, length=180,
                 command=self._on_limit_change).grid(row=0, column=1, **P)
        self.limit_lbl = tk.Label(bot, text=f'{self.limit:.1f} Nm', width=8)
        self.limit_lbl.grid(row=0, column=2, **P)
        self.fps_lbl = tk.Label(bot, text='帧率: 0 Hz', fg='gray')
        self.fps_lbl.grid(row=0, column=3, padx=16)

        # countdown + enable status
        mid = tk.Frame(self.root)
        mid.pack(fill='x', padx=10)
        self.countdown_lbl = tk.Label(mid, text='', font=('Arial', 11), fg='orange')
        self.countdown_lbl.pack(side='left')
        self.enable_lbl = tk.Label(mid, text='电机: 未使能', font=('Arial', 11, 'bold'), fg='gray')
        self.enable_lbl.pack(side='right')

        # E-stop
        tk.Button(self.root, text='⚡  紧急停止  ⚡',
                  font=('Arial', 14, 'bold'), bg='red', fg='white',
                  activebackground='#cc0000', height=2,
                  command=self._emergency_stop).pack(fill='x', padx=10, pady=8)

        self._refresh_ports()

    def _build_motor_panel(self, parent, title, side):
        frame = tk.LabelFrame(parent, text=title, padx=6, pady=6)
        rows = [('位置 (rad)', ), ('速度 (rad/s)', ), ('反馈扭矩 (Nm)', )]
        sv = []
        for i, (lbl,) in enumerate(rows):
            tk.Label(frame, text=lbl, anchor='e', width=14).grid(row=i, column=0, sticky='e')
            v = tk.StringVar(value='0.000')
            tk.Label(frame, textvariable=v, width=10, anchor='w',
                     font=('Courier', 10)).grid(row=i, column=1, sticky='w')
            sv.append(v)

        if side == 'left':
            self.lpos_v, self.lvel_v, self.ltq_v = sv
        else:
            self.rpos_v, self.rvel_v, self.rtq_v = sv

        tk.Label(frame, text='指令扭矩 (Nm)', anchor='e', width=14).grid(row=3, column=0, sticky='e')
        cmd_v = tk.DoubleVar(value=0.0)
        tk.Label(frame, textvariable=cmd_v, width=7,
                 font=('Courier', 11, 'bold')).grid(row=3, column=1)

        bf = tk.Frame(frame)
        bf.grid(row=4, column=0, columnspan=2, pady=4)
        tk.Button(bf, text='  +  ', width=4,
                  command=lambda s=side: self._adj(s, +TORQUE_STEP)).pack(side='left', padx=3)
        tk.Button(bf, text='  −  ', width=4,
                  command=lambda s=side: self._adj(s, -TORQUE_STEP)).pack(side='left', padx=3)
        tk.Button(bf, text=' 清零 ', width=4,
                  command=lambda s=side: self._set(s, 0.0)).pack(side='left', padx=3)

        if side == 'left':
            self.left_cmd_v = cmd_v
        else:
            self.right_cmd_v = cmd_v
        return frame

    # -------------------- Port --------------------
    def _refresh_ports(self):
        TEENSY_VID, TEENSY_PID = 0x16C0, 0x0483
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_cb['values'] = ports
        # 优先选中 Teensy
        for p in serial.tools.list_ports.comports():
            if p.vid == TEENSY_VID and p.pid == TEENSY_PID:
                self.port_var.set(p.device)
                return
        if ports:
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get()
        if not port:
            messagebox.showerror('错误', '请选择串口')
            return
        try:
            self.ser = serial.Serial(port, 9600, timeout=0.05)
            self.connected = True
            self._stop.clear()
            threading.Thread(target=self._read_loop,  daemon=True).start()
            threading.Thread(target=self._write_loop, daemon=True).start()
            self.btn_connect.config(text='断开')
            self.dot.config(fg='green')
            self.conn_lbl.config(text=f'已连接 {port}')
            self._start_countdown(3)
        except Exception as e:
            messagebox.showerror('连接失败', str(e))

    def _disconnect(self):
        self._safe_disable()
        self._stop.set()
        self.connected = False
        self.enabled = False
        if self.ser and self.ser.is_open:
            time.sleep(0.12)
            self.ser.close()
        self.btn_connect.config(text='连接')
        self.dot.config(fg='gray')
        self.conn_lbl.config(text='未连接')
        self.enable_lbl.config(text='电机: 未使能', fg='gray')
        self.countdown_lbl.config(text='')

    def _safe_disable(self):
        if self.enabled and self.ser and self.ser.is_open:
            try:
                self.ser.write(build_enable_frame(0xFD))
            except Exception:
                pass
            self.enabled = False

    # -------------------- Countdown --------------------
    def _start_countdown(self, n):
        self._cd = n
        self._tick()

    def _tick(self):
        if not self.connected:
            return
        if self._cd > 0:
            self.countdown_lbl.config(text=f'电机使能倒计时: {self._cd}s')
            self._cd -= 1
            self.root.after(1000, self._tick)
        else:
            self.countdown_lbl.config(text='')
            if self.ser and self.ser.is_open:
                self.ser.write(build_enable_frame(0xFC))
            self.enabled = True
            self.enable_lbl.config(text='电机: 已使能 ✓', fg='green')

    # -------------------- Serial threads --------------------
    def _write_loop(self):
        interval = 1.0 / SEND_HZ
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self.ser and self.ser.is_open:
                with self._lock:
                    frame = build_control_frame(self.seq, self.left_cmd,
                                                self.right_cmd, self.limit)
                    self.seq = (self.seq + 1) & 0xFFFF
                try:
                    self.ser.write(frame)
                except Exception:
                    pass
            dt = time.perf_counter() - t0
            rem = interval - dt
            if rem > 0:
                time.sleep(rem)

    def _read_loop(self):
        buf = b''
        while not self._stop.is_set():
            if not (self.ser and self.ser.is_open):
                time.sleep(0.01)
                continue
            try:
                chunk = self.ser.read(128)
            except Exception:
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
                self._parse(buf[:STATUS_SIZE])
                buf = buf[STATUS_SIZE:]

    def _parse(self, data: bytes):
        if data[0] != HEAD_STATUS or data[-1] != FRAME_TAIL:
            return
        crc_data = data[1:STATUS_SIZE - 2]
        if calc_crc8(crc_data) != data[-2]:
            return
        try:
            _, seq, state, err, lp, lv, lt, rp, rv, rt, ts, crc, tail = \
                struct.unpack(STATUS_FMT, data)
        except struct.error:
            return
        with self._lock:
            self.lpos, self.lvel, self.ltq = lp, lv, lt
            self.rpos, self.rvel, self.rtq = rp, rv, rt
        self._fps_count += 1
        now = time.time()
        if now - self._fps_last >= 1.0:
            self.fps = self._fps_count
            self._fps_count = 0
            self._fps_last = now

    # -------------------- Commands --------------------
    def _adj(self, side, delta):
        if side == 'left':
            self.left_cmd = round(max(-TORQUE_MAX, min(TORQUE_MAX, self.left_cmd + delta)), 2)
            self.left_cmd_v.set(self.left_cmd)
        else:
            self.right_cmd = round(max(-TORQUE_MAX, min(TORQUE_MAX, self.right_cmd + delta)), 2)
            self.right_cmd_v.set(self.right_cmd)

    def _set(self, side, val):
        if side == 'left':
            self.left_cmd = val; self.left_cmd_v.set(val)
        else:
            self.right_cmd = val; self.right_cmd_v.set(val)

    def _on_limit_change(self, _=None):
        self.limit = round(self.limit_var.get(), 1)
        self.limit_lbl.config(text=f'{self.limit:.1f} Nm')

    def _emergency_stop(self):
        with self._lock:
            self.left_cmd = 0.0
            self.right_cmd = 0.0
        self.left_cmd_v.set(0.0)
        self.right_cmd_v.set(0.0)

    # -------------------- UI refresh --------------------
    def _update_ui(self):
        with self._lock:
            lp, lv, lt = self.lpos, self.lvel, self.ltq
            rp, rv, rt = self.rpos, self.rvel, self.rtq
            fps = self.fps
        self.lpos_v.set(f'{lp:+.3f}')
        self.lvel_v.set(f'{lv:+.3f}')
        self.ltq_v.set(f'{lt:+.3f}')
        self.rpos_v.set(f'{rp:+.3f}')
        self.rvel_v.set(f'{rv:+.3f}')
        self.rtq_v.set(f'{rt:+.3f}')
        self.fps_lbl.config(text=f'帧率: {fps} Hz')
        self.root.after(50, self._update_ui)

    # -------------------- Close --------------------
    def _on_close(self):
        if self.connected:
            self._disconnect()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = MotorControlApp(root)
    app._update_ui()
    root.mainloop()
