"""
Elonxi 无线多模态设备读取器
负责通过 Elonxi_SDK.dll (pythonnet) 连接设备并接收超声数据

依赖:
    pip install pythonnet zeroconf
    DLL 路径: ../Ultra_sound_SDK/无线/Elonxi_SDK.dll
"""

import sys
import os
import time
import threading
import queue
from pathlib import Path

# 加载 .NET 运行时
from pythonnet import load
load("coreclr")
import clr

# 添加 DLL 路径
_SDK_DIR = Path(__file__).parent.parent / "Ultra_sound_SDK" / "无线"
clr.AddReference(str(_SDK_DIR / "Elonxi_SDK"))

from Elonxi_SDK import Newsletter, GlobalEvents, PacketType, SearchDevice
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf


class ElonxiReader:
    """
    Elonxi 无线超声采集器（轻量版，去除 Qt/LSL 依赖）

    用法:
        reader = ElonxiReader()
        reader.search_device()          # 自动发现设备 IP
        reader.connect("192.168.x.x")   # 或手动指定 IP
        reader.start_collection()
        # 在回调或 get_data() 中读取数据
        reader.stop_collection()
        reader.disconnect()
    """

    LOCAL_PORT = 12345
    DEVICE_PORT = 8080

    def __init__(self, ult_channels: list[int] | None = None):
        """
        :param ult_channels: 超声通道列表，例如 [0, 1]，None 则接受所有通道
        """
        self.ult_channels = ult_channels  # None = 接受全部
        self._newsletter: Newsletter | None = None

        # 线程安全队列，外部用 get_data() 取数据
        self._ult_queue: queue.Queue = queue.Queue(maxsize=500)

        self._connected = False
        self._collecting = False
        self._lock = threading.Lock()

        # 注册全局事件
        GlobalEvents.NotificationReceived += self._on_notification
        GlobalEvents.RealRealUltrDataReceived += self._on_ultrasound_data

    # ------------------------------------------------------------------
    # 设备发现
    # ------------------------------------------------------------------

    def search_device(self, timeout: float = 5.0) -> list[str]:
        """
        通过 mDNS/Zeroconf 搜索局域网内的 Elonxi 设备。
        返回发现的 IP 列表。
        """
        found_ips: list[str] = []
        event = threading.Event()

        class _Listener(ServiceListener):
            def add_service(self, zc, type_, name):
                info = zc.get_service_info(type_, name)
                if info:
                    ips = info.parsed_addresses()
                    found_ips.extend(ips)
                    print(f"[Elonxi] 发现设备: {ips}")
                    event.set()

            def remove_service(self, zc, type_, name):
                pass

            def update_service(self, zc, type_, name):
                pass

        zc = Zeroconf()
        ServiceBrowser(zc, "_http._udp.local.", _Listener())
        event.wait(timeout=timeout)
        zc.close()

        if not found_ips:
            print("[Elonxi] 未发现设备，请手动指定 IP")
        return found_ips

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------

    def connect(self, device_ip: str) -> bool:
        """连接到设备"""
        print(f"[Elonxi] 连接到 {device_ip}:{self.DEVICE_PORT} ...")
        self._newsletter = Newsletter(self.LOCAL_PORT, device_ip, self.DEVICE_PORT)
        self._newsletter.deviceSwitch(True)
        time.sleep(1.0)  # 等待连接确认
        self._connected = True
        print("[Elonxi] 连接指令已发送")
        return True

    def disconnect(self):
        """断开设备连接"""
        if self._newsletter and self._connected:
            if self._collecting:
                self.stop_collection()
            self._newsletter.deviceSwitch(False)
            self._connected = False
            print("[Elonxi] 已断开连接")

    # ------------------------------------------------------------------
    # 采集控制
    # ------------------------------------------------------------------

    def config(self, ult_channel_str: str = "0", emg_channel_str: str = "",
               imu_str: str = "", input_mod: int = 0, out_mod: int = 0,
               emg_mod: bool = False):
        """
        发送配置参数到设备。
        :param ult_channel_str: 超声通道字符串，例如 "0" 或 "0,1"
        :param emg_channel_str: EMG 通道字符串（不采集则留空）
        :param imu_str: IMU 配置字符串（不采集则留空）
        """
        if self._newsletter is None:
            raise RuntimeError("请先连接设备")
        self._newsletter.configParam(
            ult_channel_str, emg_channel_str, imu_str,
            input_mod, out_mod, emg_mod
        )
        time.sleep(0.5)

    def start_collection(self):
        """开始采集"""
        if self._newsletter is None:
            raise RuntimeError("请先连接设备")
        self._newsletter.collectionSwitch(True)
        self._collecting = True
        print("[Elonxi] 开始采集")

    def stop_collection(self):
        """停止采集"""
        if self._newsletter and self._collecting:
            self._newsletter.collectionSwitch(False)
            self._collecting = False
            print("[Elonxi] 停止采集")

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------

    def get_data(self, timeout: float = 0.1) -> tuple[float, int, list[int]] | None:
        """
        从队列取一条超声数据（非阻塞，超时返回 None）。

        返回: (timestamp_s, channel_id, waveform_list)
            timestamp_s : 系统时间戳（秒）
            channel_id  : 通道号
            waveform_list: 该通道单次回波完整波形（int 列表，长度固定）
        """
        try:
            return self._ult_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def data_available(self) -> bool:
        return not self._ult_queue.empty()

    # ------------------------------------------------------------------
    # 内部回调（由 C# 事件线程调用）
    # ------------------------------------------------------------------

    def _on_notification(self, packet_type, message):
        if packet_type == PacketType.DeviceConnection:
            print(f"[Elonxi] 设备连接: {message}")
        elif packet_type == PacketType.CollectionStatus:
            print(f"[Elonxi] 采集状态: {message}")
        elif packet_type == PacketType.BatteryCapacity:
            print(f"[Elonxi] 电量: {message}%")

    def _on_ultrasound_data(self, ultrasonic_data_by_channel):
        """
        ultrasonicDataByChannel: Dict<int, List<int[]>>
            key   : 通道号
            value : 该通道本帧包含的若干次回波（每次回波是一个 int 数组）
        """
        print(f"[DEBUG] 超声回调触发, 通道数: {len(ultrasonic_data_by_channel)}")
        ts = time.time()
        for ch, waveforms in ultrasonic_data_by_channel.items():
            if self.ult_channels is not None and ch not in self.ult_channels:
                continue
            for wf in waveforms:
                data = list(wf) if not isinstance(wf, list) else wf
                item = (ts, int(ch), data)
                try:
                    self._ult_queue.put_nowait(item)
                except queue.Full:
                    # 丢弃最旧的帧
                    try:
                        self._ult_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._ult_queue.put_nowait(item)
