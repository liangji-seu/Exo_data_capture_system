"""
XSens Awinda 无线 IMU 读取器（原始数据模式：加速度 + 角速度 + 磁力计）

依赖:
    pip install xsensdeviceapi  (从 MT SDK/Python/x64/ 下的 .whl 安装)

设备类型支持:
    - IMU: 输出 Acceleration, RateOfTurn (Gyroscope), MagneticField
    - VRU/AHRS: 输出 Quaternion + Euler angles
    - GNSS: 输出 Quaternion + LatLon + Altitude + Velocity
"""

import sys
import time
import queue
import threading

import xsensdeviceapi as xda


class XdaCallback(xda.XsCallback):
    """线程安全的数据包缓冲回调"""

    def __init__(self, max_buffer: int = 20):
        super().__init__()
        self._buffer: list = []
        self._lock = threading.Lock()
        self._max = max_buffer

    def packetAvailable(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    def getNextPacket(self) -> xda.XsDataPacket:
        with self._lock:
            assert self._buffer
            return xda.XsDataPacket(self._buffer.pop(0))

    def onLiveDataAvailable(self, dev, packet):
        with self._lock:
            while len(self._buffer) >= self._max:
                self._buffer.pop()
            self._buffer.append(xda.XsDataPacket(packet))


class XSensReader:
    """
    XSens Awinda 无线 IMU 读取器（轮询模式）

    用法:
        reader = XSensReader(sample_rate=100)
        reader.connect()          # 自动扫描端口
        reader.start()
        data = reader.get_data()  # 取一条数据，None 表示无新数据
        reader.stop()
    """

    def __init__(self, sample_rate: int = 100):
        """
        :param sample_rate: 采样率 Hz（默认100，IMU 最高支持 2000Hz）
        """
        self.sample_rate = sample_rate
        self._control: xda.XsControl | None = None
        self._device: xda.XsDevice | None = None
        self._callback: XdaCallback | None = None
        self._port_info: xda.XsPortInfo | None = None
        self._device_type: str = "unknown"  # "imu" / "vru_ahrs" / "gnss"

        # 数据队列 (timestamp, dict)
        self._data_queue: queue.Queue = queue.Queue(maxsize=500)
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 连接 / 断开
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """扫描并连接第一个找到的 XSens 设备"""
        print("[XSens] 正在扫描端口...")
        self._control = xda.XsControl_construct()
        assert self._control != 0

        port_array = xda.XsScanner_scanPorts()
        port = xda.XsPortInfo()

        for i in range(port_array.size()):
            pi = port_array[i]
            if pi.deviceId().isMti() or pi.deviceId().isMtig():
                port = pi
                break

        if port.empty():
            print("[XSens] 未找到设备，请检查 Awinda 基站连接")
            return False

        did = port.deviceId()
        print(f"[XSens] 发现设备: {did.toXsString()} @ {port.portName()}")

        if not self._control.openPort(port.portName(), port.baudrate()):
            print("[XSens] 无法打开端口")
            return False

        self._device = self._control.device(did)
        self._port_info = port
        assert self._device != 0

        # 确定设备类型
        if did.isImu():
            self._device_type = "imu"
        elif did.isVru() or did.isAhrs():
            self._device_type = "vru_ahrs"
        elif did.isGnss():
            self._device_type = "gnss"
        else:
            self._device_type = "imu"  # 默认尝试 IMU 模式
            print(f"[XSens] 未知设备类型，尝试 IMU 模式")

        print(f"[XSens] 设备类型: {self._device_type}")

        # 附加回调
        self._callback = XdaCallback()
        self._device.addCallbackHandler(self._callback)

        # 进入配置模式
        if not self._device.gotoConfig():
            print("[XSens] 无法进入配置模式")
            return False

        # 配置输出
        self._configure_output()

        # 进入测量模式
        if not self._device.gotoMeasurement():
            print("[XSens] 无法进入测量模式")
            return False

        print("[XSens] 连接成功，进入测量模式")
        return True

    def disconnect(self):
        """停止采集并关闭连接"""
        self.stop()
        if self._device and self._callback:
            self._device.removeCallbackHandler(self._callback)
        if self._control and self._port_info:
            self._control.closePort(self._port_info.portName())
            self._control.close()
        self._device = None
        self._control = None
        print("[XSens] 已断开连接")

    # ------------------------------------------------------------------
    # 采集控制
    # ------------------------------------------------------------------

    def start(self):
        """启动后台读取线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[XSens] 开始采集")

    def stop(self):
        """停止后台读取线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[XSens] 停止采集")

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------

    def get_data(self, timeout: float = 0.01) -> tuple[float, dict] | None:
        """
        从队列取一条 IMU 数据（非阻塞）。

        返回: (timestamp_s, data_dict) 或 None

        data_dict 字段（按设备类型，未提供的字段不存在）:
            acc_x, acc_y, acc_z         加速度 m/s²  (IMU)
            gyr_x, gyr_y, gyr_z         角速度 rad/s  (IMU)
            mag_x, mag_y, mag_z         磁力计 a.u.   (IMU)
            quat_w, quat_x, quat_y, quat_z  四元数   (VRU/AHRS)
            roll, pitch, yaw            欧拉角 deg    (VRU/AHRS)
            lat, lon                    经纬度        (GNSS)
            altitude                    高度 m        (GNSS)
            vel_e, vel_n, vel_u         速度 m/s      (GNSS)
        """
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def data_available(self) -> bool:
        return not self._data_queue.empty()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _configure_output(self):
        """根据设备类型配置输出数据字段"""
        cfg = xda.XsOutputConfigurationArray()
        cfg.push_back(xda.XsOutputConfiguration(xda.XDI_PacketCounter, 0))
        cfg.push_back(xda.XsOutputConfiguration(xda.XDI_SampleTimeFine, 0))

        if self._device_type == "imu":
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_Acceleration, self.sample_rate))
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_RateOfTurn, self.sample_rate))
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_MagneticField, self.sample_rate))
        elif self._device_type == "vru_ahrs":
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_Quaternion, self.sample_rate))
        elif self._device_type == "gnss":
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_Quaternion, self.sample_rate))
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_LatLon, self.sample_rate))
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_AltitudeEllipsoid, self.sample_rate))
            cfg.push_back(xda.XsOutputConfiguration(xda.XDI_VelocityXYZ, self.sample_rate))

        if not self._device.setOutputConfiguration(cfg):
            raise RuntimeError("[XSens] 配置输出失败")

    def _parse_packet(self, packet: xda.XsDataPacket) -> dict:
        """解析数据包为字典"""
        d = {}

        if packet.containsCalibratedData():
            acc = packet.calibratedAcceleration()
            d["acc_x"] = acc[0]
            d["acc_y"] = acc[1]
            d["acc_z"] = acc[2]

            gyr = packet.calibratedGyroscopeData()
            d["gyr_x"] = gyr[0]
            d["gyr_y"] = gyr[1]
            d["gyr_z"] = gyr[2]

            mag = packet.calibratedMagneticField()
            d["mag_x"] = mag[0]
            d["mag_y"] = mag[1]
            d["mag_z"] = mag[2]

        if packet.containsOrientation():
            q = packet.orientationQuaternion()
            d["quat_w"] = q[0]
            d["quat_x"] = q[1]
            d["quat_y"] = q[2]
            d["quat_z"] = q[3]

            euler = packet.orientationEuler()
            d["roll"]  = euler.x()
            d["pitch"] = euler.y()
            d["yaw"]   = euler.z()

        if packet.containsLatitudeLongitude():
            ll = packet.latitudeLongitude()
            d["lat"] = ll[0]
            d["lon"] = ll[1]

        if packet.containsAltitude():
            d["altitude"] = packet.altitude()

        if packet.containsVelocity():
            vel = packet.velocity(xda.XDI_CoordSysEnu)
            d["vel_e"] = vel[0]
            d["vel_n"] = vel[1]
            d["vel_u"] = vel[2]

        return d

    def _read_loop(self):
        """后台线程：轮询回调缓冲，将数据入队"""
        while self._running:
            if self._callback and self._callback.packetAvailable():
                packet = self._callback.getNextPacket()
                ts = time.time()
                data = self._parse_packet(packet)
                if data:
                    item = (ts, data)
                    try:
                        self._data_queue.put_nowait(item)
                    except queue.Full:
                        try:
                            self._data_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._data_queue.put_nowait(item)
            else:
                time.sleep(0.001)  # 1ms 空转避免CPU飙升
