# IMU + 超声同步采集系统

## 依赖安装

### 1. XSens IMU (xsensdeviceapi)
根据你的 Python 版本选择对应的 .whl 文件安装：
```bash
# Python 3.9 + Windows x64（示例）
pip install "../MT SDK/Python/x64/xsensdeviceapi-2025.2.0-cp39-none-win_amd64.whl"
```

### 2. Elonxi SDK 依赖
```bash
pip install pythonnet zeroconf
```

> 注意：Elonxi_SDK.dll 路径硬编码在 `elonxi_reader.py` 中，
> 指向 `../Ultra_sound_SDK/无线/Elonxi_SDK.dll`，无需额外配置。

---

## 运行

```bash
cd Capture_System

# 自动搜索 Elonxi 设备，采集 60 秒
python main_capture.py --duration 60

# 手动指定设备 IP，两个超声通道，IMU 200Hz
python main_capture.py --device-ip 192.168.1.100 --ult-channels 0 1 --imu-rate 200

# 持续采集直到 Ctrl+C
python main_capture.py --ult-channels 0
```

---

## 输出文件

采集完成后在 `./data/` 目录下生成两个 CSV 文件：

### `imu_YYYYMMDD_HHMMSS.csv`
| 列名 | 说明 |
|------|------|
| timestamp | Unix 时间戳（秒，6位小数） |
| acc_x/y/z | 加速度 m/s² |
| gyr_x/y/z | 角速度 rad/s |
| mag_x/y/z | 磁力计 |
| roll/pitch/yaw | 欧拉角 deg（VRU/AHRS 设备才有） |
| quat_w/x/y/z | 四元数（VRU/AHRS 设备才有） |

### `ult_YYYYMMDD_HHMMSS.csv`
| 列名 | 说明 |
|------|------|
| timestamp | Unix 时间戳（秒，6位小数） |
| channel | 超声通道号 |
| pt0, pt1, ..., ptN | 单次回波完整波形，每个点对应固定深度的回波幅度（int） |

---

## 关于超声数据的含义

A 型超声的工作原理：发射超声脉冲 → 等待组织界面反射的回波 → ADC 采样回波信号。

因此 `pt0~ptN` 是一条**时间轴上的回波幅度序列**：
- 下标越大 = 声波传播时间越长 = 探测深度越深
- 值的大小 = 该深度处的回波强度（界面反射越强，值越大）
- 数组长度固定（由设备配置决定，首次接收时自动确定并写入 CSV 列头）

---

## 代码结构

```
Capture_System/
├── main_capture.py     # 主入口，命令行参数，采集主循环，CSV 写入
├── elonxi_reader.py    # Elonxi 无线超声/EMG 设备读取器
└── xsens_reader.py     # XSens Awinda 无线 IMU 读取器
```
