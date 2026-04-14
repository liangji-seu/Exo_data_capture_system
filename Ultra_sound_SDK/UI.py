

import sys
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

# 导入你提供的核心模块
from elonxiPy_1 import NewsletterBase, EMGFilter, DeviceSearcher


# -------------------------- 实时绘图组件 --------------------------
class EMGPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        # 创建matplotlib画布
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = self.fig.add_subplot(211)  # 通道1
        self.ax2 = self.fig.add_subplot(212)  # 通道2
        self.fig.tight_layout()

        super(EMGPlotCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # 初始化数据缓存（固定长度，实现滚动显示）
        self.buffer_len = 500  # 显示500个采样点
        self.ch1_data = np.zeros(self.buffer_len)
        self.ch2_data = np.zeros(self.buffer_len)

        # 初始化绘图线条
        self.line1, = self.ax1.plot(range(self.buffer_len), self.ch1_data, 'b-')
        self.line2, = self.ax2.plot(range(self.buffer_len), self.ch2_data, 'r-')

        # 设置坐标轴
        self.ax1.set_title('EMG Channel 1')
        self.ax1.set_ylim(-1000, 1000)
        self.ax1.set_xlim(0, self.buffer_len)
        self.ax2.set_title('EMG Channel 2')
        self.ax2.set_ylim(-1000, 1000)
        self.ax2.set_xlim(0, self.buffer_len)

    def update_plot(self, ch1_val=None, ch2_val=None):
        """更新绘图数据（滚动显示）"""
        # 更新通道1数据
        if ch1_val is not None:
            self.ch1_data = np.roll(self.ch1_data, -1)
            self.ch1_data[-1] = ch1_val
            self.line1.set_ydata(self.ch1_data)   #更新通道曲线

        # 更新通道2数据
        if ch2_val is not None:
            self.ch2_data = np.roll(self.ch2_data, -1)
            self.ch2_data[-1] = ch2_val
            self.line2.set_ydata(self.ch2_data)

        # 刷新画布
        self.draw()


# -------------------------- 主UI窗口 --------------------------
class EMGCollectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('肌电信号采集系统 (通道1/2)')
        self.setGeometry(100, 100, 1000, 800)  #窗口位置和大小

        # 核心变量初始化
        self.newsletter = None  # 通信核心对象，为什么初始未连接
        self.emg_filter = EMGFilter()  # EMG滤波器
        self.collected_data = {1: [], 2: []}  # 存储采集的原始/滤波数据
        self.is_collecting = False  # 采集状态标记
        self.save_file_path = ""  # 数据保存路径

        # 创建UI组件
        self._init_ui()  #在下面进行定义

        # 初始化设备搜索
        self._init_device_search()  #初始化设备搜索

    def _init_ui(self):
        """初始化UI布局"""
        # 中心窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)  #设置中心部件
        main_layout = QVBoxLayout(central_widget)  #中心部件的主布局（垂直布局）

        # 1. 设备连接区域
        connect_group = QGroupBox("设备连接")
        connect_layout = QHBoxLayout(connect_group)

        self.lbl_local_port = QLabel("本地端口:")
        self.edit_local_port = QLineEdit("1430")  # 默认本地端口
        self.lbl_remote_ip = QLabel("设备IP:")
        self.edit_remote_ip = QLineEdit("192.168.31.203")  # 默认设备IP
        self.lbl_remote_port = QLabel("设备端口:")
        self.edit_remote_port = QLineEdit("1430")  # 默认设备端口
        self.btn_connect = QPushButton("连接设备")
        self.btn_disconnect = QPushButton("断开连接")
        self.btn_disconnect.setEnabled(False)  #初始化禁用断开按钮

        connect_layout.addWidget(self.lbl_local_port)
        connect_layout.addWidget(self.edit_local_port)
        connect_layout.addWidget(self.lbl_remote_ip)
        connect_layout.addWidget(self.edit_remote_ip)
        connect_layout.addWidget(self.lbl_remote_port)
        connect_layout.addWidget(self.edit_remote_port)
        connect_layout.addWidget(self.btn_connect)
        connect_layout.addWidget(self.btn_disconnect)

        # 2. 采集控制区域
        control_group = QGroupBox("采集控制")
        control_layout = QHBoxLayout(control_group)  #水平布局

        self.btn_start = QPushButton("开始采集")
        self.btn_stop = QPushButton("停止采集")
        self.btn_save = QPushButton("保存数据")
        self.btn_start.setEnabled(False)  #初始禁用
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_save)

        # 3. 实时绘图区域
        plot_group = QGroupBox("实时肌电波形 (通道1/2)")
        plot_layout = QVBoxLayout(plot_group)
        self.plot_canvas = EMGPlotCanvas(self, width=9, height=5)
        plot_layout.addWidget(self.plot_canvas)

        # 4. 数据显示区域
        data_group = QGroupBox("原始数据显示 (最新100条)")
        data_layout = QVBoxLayout(data_group)
        self.txt_data = QTextEdit()
        self.txt_data.setReadOnly(True)  #文本编辑框设置为只读
        data_layout.addWidget(self.txt_data)

        # 添加所有组到主布局
        main_layout.addWidget(connect_group)
        main_layout.addWidget(control_group)
        main_layout.addWidget(plot_group)
        main_layout.addWidget(data_group)

        # 绑定按钮事件
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        self.btn_start.clicked.connect(self._on_start_collect)
        self.btn_stop.clicked.connect(self._on_stop_collect)
        self.btn_save.clicked.connect(self._on_save_data)

    def _init_device_search(self):
        """初始化设备搜索（自动发现局域网内设备）"""
        self.device_searcher = DeviceSearcher()
        # 绑定设备搜索信号（找到设备后自动填充IP）
        self.device_searcher.sendSearcherSig.connect(self._on_device_found)

    # -------------------------- 信号槽函数 --------------------------
    def _on_device_found(self, ip_list):
        """设备搜索回调：自动填充设备IP"""
        if ip_list:
            self.edit_remote_ip.setText(ip_list[0])
            QMessageBox.information(self, "设备发现", f"找到设备IP: {ip_list[0]}")

    def _on_connect(self):
        """连接设备"""
        try:
            # 获取连接参数
            local_port = int(self.edit_local_port.text())
            remote_ip = self.edit_remote_ip.text()
            remote_port = int(self.edit_remote_port.text())

            # 初始化通信核心对象
            self.newsletter = NewsletterBase(local_port, remote_ip, remote_port)
            # 绑定EMG数据接收信号
            self.newsletter.sendData_Sig.connect(self._on_emg_data_received1)
           # self.newsletter.emgData_Sig.connect(self._on_emg_data_received2)
            # 连接设备
            self.newsletter.deviceSwitch(True)

            # 更新按钮状态
            self.btn_connect.setEnabled(False) #禁用连接按钮
            self.btn_disconnect.setEnabled(True)  #启用断开按钮
            self.btn_start.setEnabled(True)  #启用开始采集按钮
            QMessageBox.information(self, "成功", "设备连接成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"设备连接失败：{str(e)}")

    def _on_disconnect(self):
        """断开设备"""
        if self.newsletter:
            # 停止采集（如果正在采集）
            if self.is_collecting:
                self._on_stop_collect()
            # 断开连接
            self.newsletter.deviceSwitch(False)
            self.newsletter = None

            # 更新按钮状态
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            self.btn_start.setEnabled(False)
            QMessageBox.information(self, "成功", "设备已断开连接！")

    def _on_start_collect(self):
        """开始采集"""
        if not self.newsletter:
            QMessageBox.warning(self, "警告", "请先连接设备！")
            return

        try:
            # 清空历史数据
            self.collected_data = {1: [], 2: []}
            self.txt_data.clear()

            # 配置采集参数（仅启用通道1/2的EMG）
            self.newsletter.configParam(
                ultr="",  # 关闭超声
                emg="1",  # 启用EMG通道1/2
                imu="",  # 关闭IMU
                inputMod=0,  # 根据实际需求调整输入模式
                outMod=0,  # 根据实际需求调整输出模式
                emgMod=False  # 根据实际需求调整EMG模式
            )

            # 启动采集
            self.newsletter.collectionSwitch(True)
            self.newsletter.setStartSaveFile(True)
            self.is_collecting = True

            # 更新按钮状态
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_save.setEnabled(False)
            QMessageBox.information(self, "成功", "开始采集肌电信号！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"采集启动失败：{str(e)}")

    def _on_stop_collect(self):
        """停止采集"""
        if not self.newsletter or not self.is_collecting:
            return

        try:
            # 停止采集
            self.newsletter.collectionSwitch(False)
            self.newsletter.setStartSaveFile(False)
            self.is_collecting = False

            # 更新按钮状态
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_save.setEnabled(True)
            QMessageBox.information(self, "成功", "采集已停止！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"采集停止失败：{str(e)}")

    def _on_save_data(self):
        """保存采集数据"""
        if not self.collected_data[1] and not self.collected_data[2]:
            QMessageBox.warning(self, "警告", "暂无采集数据可保存！")
            return

        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", f"EMG数据_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;CSV文件 (*.csv)"
        )
        if not file_path:
            return

        try:
            # 保存数据（格式：时间戳,通道1,通道2）
            max_len = max(len(self.collected_data[1]), len(self.collected_data[2]))
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("时间戳,通道1,通道2\n")
                for i in range(max_len):
                    ts = time.time()  # 可替换为实际采集时间戳
                    ch1 = self.collected_data[1][i] if i < len(self.collected_data[1]) else ""
                    ch2 = self.collected_data[2][i] if i < len(self.collected_data[2]) else ""
                    f.write(f"{ts},{ch1},{ch2}\n")

            QMessageBox.information(self, "成功", f"数据已保存至：{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据保存失败：{str(e)}")

    # @pyqtSlot(int, list, bool, int)
    # def _on_emg_data_received(self, channel, data_list, is_emg, pack_num):
    #     """接收EMG数据信号槽函数"""
    #     # 仅处理通道1/2的EMG数据
    #     if not is_emg or channel not in [1, 2]:
    #         return
    #
    #     # 滤波处理
    #     filtered_data = self.emg_filter.filter_signal(np.array(data_list))
    #
    #     # 存储数据（取均值简化显示，也可存储原始数组）
    #     avg_val = np.mean(filtered_data)
    #     self.collected_data[channel].append(avg_val)
    #
    #     # 更新实时绘图（取最后一个滤波值）
    #     if channel == 1:
    #         self.plot_canvas.update_plot(ch1_val=filtered_data[-1], ch2_val=None)
    #     elif channel == 2:
    #         self.plot_canvas.update_plot(ch1_val=None, ch2_val=filtered_data[-1])
    #
    #     # 更新文本显示（仅保留最新100条）
    #     display_text = self.txt_data.toPlainText()
    #     new_line = f"包编号:{pack_num} | 通道{channel} | 原始均值:{np.mean(data_list):.2f} | 滤波均值:{avg_val:.2f}\n"
    #     if len(display_text.splitlines()) >= 100:
    #         display_text = "\n".join(display_text.splitlines()[1:])
    #     self.txt_data.setText(display_text + new_line)
    #     # 滚动到最后一行
    #     self.txt_data.moveCursor(self.txt_data.textCursor().End)

    @pyqtSlot(int, list, bool, int)   #用于明确指定该函数接收的信号参数类型
    def _on_emg_data_received1(self, channel, data_list, is_emg, pack_num):
        """接收EMG数据信号槽函数（完善版）"""
        # 仅处理通道1/2的EMG数据
        # if not is_emg or channel not in [1, 2]:
        #     return
        # 调试：打印接收到的数据（确认是否有数据传入）
        print(f"通道{channel}数据：{data_list}")
        print(is_emg)
        return
        # 1. 存储数据到collected_data
        self.collected_data[channel].extend(data_list)

        # 2. 更新实时绘图（取最后一个数据点更新，或批量更新）
        # 若data_list是批量数据，取最后一个点更新波形
        if data_list:
            last_data = data_list[-1]
            if channel == 1:
                self.plot_canvas.update_plot(ch1_val=last_data)
            else:
                self.plot_canvas.update_plot(ch2_val=last_data)

        # 3. 更新数据显示文本框（仅显示最新100条）
        data_str = f"通道{channel}：{data_list}\n"
        current_text = self.txt_data.toPlainText()
        if len(current_text.splitlines()) >= 100:
            # 超过100行则清空旧数据
            self.txt_data.clear()
        self.txt_data.insertPlainText(data_str)
        # 滚动到底部
        self.txt_data.moveCursor(self.txt_data.textCursor().End)


# 肌电专属槽函数（仅处理肌电数据，无需过滤）
    @pyqtSlot(int, list, int)  # 参数：通道、肌电数据、包号
    def _on_emg_data_received2(self, channel, data_list, pack_num):
        # 仅处理通道1/2（你关注的肌电通道）
        if channel not in [1, 2]:
            return

        # 1. 调试打印（确认信号触发）
        print(f"【肌电专属信号】通道{channel}数据：{data_list}")

        # 2. 存储数据
        self.collected_data[channel].extend(data_list)

        # 3. 更新实时绘图
        if data_list:
            last_data = data_list[-1]
            if channel == 1:
                self.plot_canvas.update_plot(ch1_val=last_data)
            else:
                self.plot_canvas.update_plot(ch2_val=last_data)

        # 4. 更新文本框显示
        data_str = f"【肌电】通道{channel}：{data_list}\n"
        current_text = self.txt_data.toPlainText()
        if len(current_text.splitlines()) >= 100:
            self.txt_data.clear()
        self.txt_data.insertPlainText(data_str)
        self.txt_data.moveCursor(self.txt_data.textCursor().End)


# # 可选：通用信号槽函数（仅处理超声数据）
# @pyqtSlot(int, list, bool, int)
# def _on_general_data_received(self, channel, data_list, is_emg, pack_num):
#     # 只处理超声数据（is_emg=False）
#     if not is_emg:
#         print(f"【超声信号】通道{channel}数据：{data_list}")
#         # 如需显示超声数据，可在这里添加逻辑

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EMGCollectUI()
    window.show()
    sys.exit(app.exec_())