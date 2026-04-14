# 核心功能是和硬件设备通信、处理传感器数据、处理丢包补包、数据滤波、LSL（Lab Streaming Layer）数据推送等
#------------------------------------一、核心依赖与初始化-------------------------------------------
#------------------------------------1、关键库引入-------------------------------------------
from PyQt5.QtCore import QObject, pyqtSignal, QTimer #用于 Qt 信号槽(基于事件驱动的机制)机制（跨线程通信）、定时器（QTimer）；
from pythonnet import load  #实现 Python 调用 C# 的 Elonxi_SDK；
from PyQt5.QtWidgets import QApplication
load("coreclr")  ## 加载.NET Core运行时（coreclr），为调用C# SDK做准备
import queue  #队列模块，用于线程安全的数据缓存
import clr  # clr模块：pythonnet核心模块，用于加载C#程序集和类型
#import copy
import asyncio # 异步IO模块：异步任务处理（本代码暂未直接使用）
#import numpy as np # 数值计算模块：数组处理、数值运算（如EMG/IMU数据处理）
import time # 时间模块：延时、时间戳处理
import threading # 线程模块：创建后台线程（如数据绘图更新线程）
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf #设备发现（基于 mDNS 的局域网设备搜索）；
import socket
clr.AddReference('System.Collections')  # 加载C#的System.Collections程序集（提供泛型集合支持）
clr.AddReference("./无线/Elonxi_SDK") # 加载本地的Elonxi_SDK.dll（硬件通信核心SDK）
#------------------------------------2.全局核心对象-------------------------------------------
from Elonxi_SDK import Newsletter,GlobalEvents,PacketType
#Newsletter（C# SDK 类）：负责和硬件设备的网络通信（指定本地端口、远端 IP / 端口）；
#GlobalEvents（C# SDK 事件）：订阅设备的各类数据 / 状态事件（如 EMG 数据接收、丢包通知、IMU 数据接收等）。
from System.Collections.Generic import List  # 导入C#泛型List类（用于和C# SDK交互时的集合类型转换）

from scipy.signal import butter, filtfilt #Butterworth 滤波器（肌电数据滤波）；
from LSLProtocol import SendSLStream  # LSL协议模块：用于将传感器数据推送到LSL流（供其他程序实时接收）
import copy  # 深拷贝模块：用于数据副本创建，避免原数据被修改
#------------------------------------二、核心类-------------------------------------------
#------------------------------------1.肌电数据滤波类-------------------------------------------
class EMGFilter:
    """lo
    肌电数据需要经过两个滤波步骤，
    一是20-450Hz带通滤波，滤波器类型为4阶Butterworth带通滤波器，
    二是工频陷波，陷波带48-52Hz，2阶Butterworth陷波滤波器，
    在线的时候只对50Hz进行陷波，
    离线的时候还对50Hz的倍频（50、100、150、200、250、300、350、400、450）也进行了陷波
    """
    def __init__(self):
        """
        初始化EMGFilter类。
        :param fs: 采样频率 (Hz)
        """
        self.fs = 1000
        self.b_bandpass, self.a_bandpass = self.butter_bandpass(20, 450, order=4)
        self.b_bandstop, self.a_bandstop = self.butter_bandstop(48, 52, order=2)

    def butter_bandpass(self, lowcut, highcut, order=4):
        """
        设计Butterworth带通滤波器
        :param lowcut: 带通下限频率（Hz）
        :param highcut: 带通上限频率（Hz）
        :param order: 滤波器阶数
        :return: 滤波器系数(b, a)
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq  #将频率归一化
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band') #设计巴特沃斯带通滤波器
        return b, a

    def butter_bandstop(self, lowcut, highcut, order=2):
        """
        设计Butterworth陷波滤波器（带阻滤波器）
        :param lowcut: 陷波下限频率（Hz）
        :param highcut: 陷波上限频率（Hz）
        :param order: 滤波器阶数
        :return: 滤波器系数(b, a)
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a

    def filter_signal(self, data):
        """
        在线EMG数据滤波（实时采集时使用）
        :param data: 原始EMG数据数组
        :return: 滤波后的EMG数据
        """
        #print(len(data))
        # 第一步：带通滤波（20-450Hz），padlen=数据长度-1确保滤波无边界失真
        data = filtfilt(self.b_bandpass, self.a_bandpass, data,padlen=len(data)-1)
        # 第二步：陷波滤波（48-52Hz），去除50Hz工频干扰
        data = filtfilt(self.b_bandstop, self.a_bandstop, data,padlen=len(data)-1)
        return data

    def filter_signal_offline(self, data, harmonics):
        """
        离线EMG数据滤波（含50Hz倍频陷波）
        :param data: 原始EMG数据数组
        :param harmonics: 50Hz的倍频值（如50/100/150...450）
        :return: 滤波后的EMG数据
        """
        # 针对指定倍频值，设计±2Hz的陷波滤波器（如50Hz则48-52Hz）
        b, a = self.butter_bandstop(harmonics - 2, harmonics + 2, order=2)
        data = filtfilt(self.b_bandpass, self.a_bandpass, data)
        data = filtfilt(b, a, data)
        return data

#------------------------------------2.核心数据通信类（继承QObject）-------------------------------------------
class NewsletterBase(QObject):
    # Qt信号定义（跨线程通信）：接收设备状态/通知信号（参数：PacketType枚举、消息内容）
    receiveSignal = pyqtSignal(object, object) #接收设备状态/通知信号（参数：PacketType枚举、消息内容）
    emgSheddSignal = pyqtSignal(object) #EMG数据丢包通知信号（参数：丢包列表）
    sendData_Sig = pyqtSignal(int, list , bool , int) #数据发送信号（参数：通道号、数据列表、是否EMG数据、包编号）
    emgData_Sig = pyqtSignal(int, list, int)  # 通道、肌电数据列表、包号（参数按需调整）
    IMUData_send_Sig = pyqtSignal(dict) #IMU数据发送信号（参数：IMU数据字典）
    pkg_doneSignal = pyqtSignal(bool, object, object, object) #补包完成信号（参数：是否完成、是否丢包严重、EMG补包列表、超声补包列表）
    pkgMaxCountSignal = pyqtSignal(object, object) #补包总数信号（参数：数据类型(ult/emg)、最大补包数）
    pkgNowCountSignal = pyqtSignal(object) #当前补包数信号（参数：已补包数）
    def __init__(self,localPort,remoteAddress,remotePort):
        """
        初始化核心通信类
        :param localPort: 本地通信端口
        :param remoteAddress: 设备远端IP地址
        :param remotePort: 设备远端端口
        """
        super(NewsletterBase, self).__init__()
        # self._filter = FilterProcessor()
        print(remoteAddress)
        self.MAX_CREATE_VIRTUAL_PER = 0.05 #创建虚假数据的最大比例,丢包数大于这个比例,则不创建虚假数据
        # 初始化C# SDK的Newsletter对象，建立本地-设备的网络通信
        self._newsletter = Newsletter(localPort,remoteAddress,remotePort)
        # 丢包字典：存储EMG丢包数据
        self.sheddDict = {}
        # EMG数据缓冲：用于排序和批量处理
        self._waveBuffer = []

        self._ultrData = []  # 若需要缓存超声数据，初始化为空列表
        self._confUltChannels = []  # 超声通道配置，初始化为空列表
        # 补包标记：录制结束后开始补包则设为True
        self.isReplenishedPackage = False
        # 实时记录超声数据最新检测到的包编号
        self._currUltPackNumber = 0
        # 实时记录EMG数据最新检测到的丢失包编号
        self._currEmgPackNumber = 0
        # 最大EMG包编号（用于缓冲排序），即最近的那个包编号
        self._maxEmgPackNumber = 0
        # 丢包列表：key=ult(超声)/emg(肌电)，value=丢包包编号列表
        self._lostPackets = {'ult': [], 'emg': []}
        # self.warning = False
        self._data_queue = queue.Queue() #线程安全队列：存储待处理的EMG数据（避免主线程阻塞）
        self._isStartSaveFile = False  # 保存文件标记：是否开始保存数据（采集启动后设为True）
        self.max_retries = 10  # 限制最大递归次数，防止无限递归
        self.current_retry = 0  # 当前递归深度
        self.initSheddDict()  # 初始化丢包字典
        self.initLSL()     # 初始化LSL数据流（用于推送EMG/IMU/超声数据）

        # 待补包列表：超声/EMG丢包包编号
        self.missing_ult = []
        self.missing_emg = []
        self.m_supPackageMark = True # 补包强行中断的标志位, False 时强制终端循环
        # self.m_alreadySupPack = 0 # 第二次进入递归开始, 记录的已经补包的总包数
        self.m_emgSupPackCount = 0  # EMG补包计数：已补EMG包数量
        self.m_ultSupPackCount = 0
        self.m_nowSupPackCount = 0  # 当前补包计数：本轮补包已处理数量
        self.m_maxEmgPackCount = 0  # 最大EMG补包数：本轮需要补的EMG总包数
        self.m_maxUltPackCount = 0
        self.m_supRare = 10         #补包频率，暂未使用

        self.m_lastEmgSupCount = 0  # 上一轮EMG补包数：用于判断是否有新的丢包
        self.m_lastUltSupCount = 0

        self.m_ConfigMaxPackCount = {} #从配置中获取到的, 包最多能有多少个  从配置文件读取的超声/EMG最大包数（防止补包越界）

        # EMG补包定时器：定时触发EMG补包逻辑（1秒间隔）
        self.m_emgSupTimer = QTimer()
        self.m_emgSupTimer.setInterval(1000)  #定时器触发间隔
        self.m_emgSupTimer.timeout.connect(self.emgPackageTimeOutFunction) #间隔到时，自动发射信号，槽函数自动调用

        # 超声补包定时器：定时触发超声补包逻辑（1秒间隔）
        self.m_ultSupTimer = QTimer()
        self.m_ultSupTimer.setInterval(1000)
        self.m_ultSupTimer.timeout.connect(self.ultrPackageTimeOutFunction)

        # |订阅|GlobalEvents的事件（在SDK中申明）
        GlobalEvents.NotificationReceived += self.on_notification_received   #  设备通知
        GlobalEvents.RealRealUltrDataReceived += self.on_real_ultr_data_received  #  超声数据
        GlobalEvents.RealRealRelDataReceived += self.on_real_RelData_Received  #  丢包通知(或许是获取实时包编号？)
        GlobalEvents.RealRealEMGReceived += self.on_real_emg_received  #  EMG数据
        # GlobalEvents.RealRealEMGReceived += self.trygogoo
        GlobalEvents.RealRealEmgSheddingReceived += self.on_real_emg_shedding_received  #  EMG丢包
        GlobalEvents.RealReaIMUReceived += self.on_real_imu_received  # IMU数据

        # 启动后台线程：更新数据绘图（守护线程，随主线程退出）
        threading.Thread(target=self.update_plot, daemon=True).start()

    def initLSL(self):
         """
         初始化LSL数据流推送对象
         """
         self._emgLSL = SendSLStream('emg',8,100)
         self._imuLSL = SendSLStream('imu', 36, 100)

    def setConfUltChannels(self,confUltChannels):
        """
       设置超声通道配置
       :param confUltChannels: 超声通道列表（如[0,1,2]）
        """
        self._confUltChannels = confUltChannels
        self._ultrData = [] #超声波数据缓冲
        self._ultrLSL = SendSLStream('ultr', len(confUltChannels), 1000)

    def initSheddDict(self):
        """初始化丢包字典（清空EMG丢包列表）"""
        self.sheddDict.clear()
        self.sheddDict["emg"] = []

    # def setWarning(self, state):
    #     self.warning = state

#------------------------------------ 事件处理函数-------------------------------
    def on_notification_received(self, onType, message):
        """
        设备通知事件回调（处理各类设备状态通知）
        :param onType: PacketType枚举（通知类型）
        :param message: 通知内容（字符串/布尔/数值）
        """
        if(onType == PacketType.DeviceConnection):  #输出在控制台，用于调试
            # 连接成功
            self.initSheddDict() #连接成功时重新初始化丢包字典
            print(f"deviceConnect info:{message}")
        elif(onType == PacketType.Configuration):
            # 配置下发成功
            print(f"config param info:{message}")
        elif(onType == PacketType.CollectionStatus):
            # 采集状态通知（启动/停止）
            print(f"collection status info:{message}")
        elif(onType == PacketType.IsPCStopMark):
            #设备由谁停止（停止来源通知）, True 是上位机停止, False 是下位机停止
            print(f"stop mark:{message}")
        elif(onType == PacketType.BatteryCapacity):
            # 电池容量
            print(f"battery capecityLa:{message}")
        elif(onType == PacketType.IsDeviceOnline):
            # 设备是否在线
            print(f"Is device online:{message}")
        elif(onType == PacketType.TriggerTimestamp):
            # 触发时间戳通知
            print(f"Trigger Timestamp la:{message}")

        # 发送Qt信号：将通知转发到UI线程处理
        self.receiveSignal.emit(onType, message)

    def on_real_RelData_Received(self , isUlt , packNumber):
        """
        丢包通知事件回调（记录丢失的包编号）（或许是获取实时包编号）
        :param isUlt: 是否超声数据（True=超声，False=EMG）
        :param packNumber: 丢失的包编号（或许是接收到的编号）
        """
        #print(isUlt , packNumber)
        if isUlt:
            self._currUltPackNumber = packNumber
        else:
            self._currEmgPackNumber = packNumber

        if not self.getStartSaveFile():
            return

        #print("accpte: ", isUlt, packNumber)
        if isUlt:
            self._lostPackets['ult'].append(packNumber)
        else:
            self._lostPackets['emg'].append(packNumber)

    def lostPacketSumNum(self):
        """预留方法：统计丢包总数（暂未实现）"""
        pass
    def setStartSaveFile(self,state):
        """
        设置数据保存标记（采集启动/停止时调用）
        :param state: True=开始保存，False=停止保存
        """
        self._isStartSaveFile = state
    def getStartSaveFile(self):
        """获取数据保存标记状态"""
        return self._isStartSaveFile
    
    def setSupPackageMark(self, mark):
        """
        设置补包强制中断标记
        :param mark: True=允许补包，False=强制中断补包
        """
        self.m_supPackageMark = mark
        #重置补包计数
        self.m_nowSupPackCount = 0
        self.m_emgSupPackCount = 0
        self.m_ultSupPackCount = 0

        self.m_lastEmgSupCount = 0
        self.m_lastUltSupCount = 0

    def initMaxPackCount(self, countCfg):
        """
        初始化配置最大包数（防止补包越界）
        :param countCfg: 字典，key=ult/emg，value=最大包数
        """
        self.m_ConfigMaxPackCount = countCfg

    def checkMissingListIsOver(self):
        """检查待补包列表是否超出配置最大包数，超出则截断"""
        # 处理超声待补包列表
        if "ult" in self.m_ConfigMaxPackCount.keys():
            # 条件1：先判断是否配置了超声最大包数（避免key不存在报错）
            if len(self.missing_ult) > self.m_ConfigMaxPackCount["ult"]:
                # missing_ult是一个存储超声丢失数据包编号的列表
                # 条件2：如果待补包列表长度 > 超声最大包数（列表过长，需要裁剪）
                if self.m_ultSupPackCount < self.m_ConfigMaxPackCount["ult"]:
                    # 子条件1：已补包数 < 最大包数（还在合法补包范围内）
                    self.missing_ult = self.missing_ult[self.m_ultSupPackCount:self.m_ConfigMaxPackCount["ult"]]
                else:
                    # 子条件2：已补包数 ≥ 最大包数（补包已超上限，直接裁剪到最大包数内）
                    self.missing_ult = self.missing_ult[0:self.m_ConfigMaxPackCount["ult"]]

        #处理EMG待补包列表
        if "emg" in self.m_ConfigMaxPackCount.keys():
            if len(self.missing_emg) > self.m_ConfigMaxPackCount["emg"]:
                if self.m_emgSupPackCount < self.m_ConfigMaxPackCount["emg"]:
                    self.missing_emg = self.missing_emg[self.m_emgSupPackCount:self.m_ConfigMaxPackCount["emg"]]
                else:
                    self.missing_emg = self.missing_emg[0:self.m_ConfigMaxPackCount["emg"]]

    def find_discontinuities(self):
        """
        核心补包逻辑：查找丢包包编号的不连续区间，触发补包请求
        递归调用自身，直到补包完成/中断/达到最大重试次数
        """
        def sub(nums):
            """
            子函数：查找有序列表中的不连续区间（即丢失的包编号）
            :param nums: 已接收的包编号列表（去重+排序后）
            :return: 丢失的包编号列表
            """
            nums = sorted(set(nums)) #去重和排序
            missing = []
            for i in range(len(nums) - 1):
                if nums[i + 1] - nums[i] > 1:
                    missing.extend(range(nums[i] + 1, nums[i + 1]))
            return missing
        
        self.m_nowSupPackCount = 0 #  重置当前补包计数
        self.current_retry += 1  #  递归深度加1
        #开始做补包处理
        #self.collectionSwitch(False)
        time.sleep(0.1)  # 短暂延时，确保数据状态稳定
        #QApplication.processEvents()
        #  深拷贝丢包列表（避免原列表被修改）
        ult = copy.deepcopy(self._lostPackets['ult'])
        emg = copy.deepcopy(self._lostPackets['emg'])
        self.missing_ult = sub(ult)
        self.missing_emg = sub(emg)

        # 标记：超声/EMG补包是否无新进展（上一轮和本轮丢包数一致）
        ultStopMark = False
        emgStopMark = False

        # 检查超声补包是否无新进展
        if self.m_lastUltSupCount != len(self.missing_ult):
            self.m_lastUltSupCount = len(self.missing_ult)
        else:
            ultStopMark = True

        # 检查EMG补包是否无新进展
        if self.m_lastEmgSupCount != len(self.missing_emg):
            self.m_lastEmgSupCount = len(self.missing_emg)
        else:
            emgStopMark = True

        # 终止补包条件：
        # 1. 超声和EMG补包均无新进展；2. 强制中断补包；3. 达到最大递归次数
        if (ultStopMark and emgStopMark) or self.m_supPackageMark==False or (self.max_retries < self.current_retry):
            loseSerious = False  # 丢包是否严重（超过最大虚假数据比例）
            sendEmg = []         # 需创建虚假数据的EMG包列表
            sendUlt = []         # 需创建虚假数据的超声包列表

            # self.m_supPackageMark 为False 是停止补包,因此不进行数据造假
            if self.m_supPackageMark:
                # 计算EMG丢包比例
                emgMaxCount = len(self._lostPackets['emg'])
                ultMaxCount = len(self._lostPackets['ult'])
                # EMG丢包比例≤阈值时，创建虚假数据
                if emgMaxCount != 0:
                    if len(self.missing_emg)/emgMaxCount <= self.MAX_CREATE_VIRTUAL_PER:
                        sendEmg = copy.deepcopy(self.missing_emg)
                    else:
                        loseSerious = True   # 丢包严重，不创建虚假数据

                # 超声丢包比例≤阈值时，创建虚假数据
                if ultMaxCount != 0:
                    if len(self.missing_ult)/ultMaxCount <= self.MAX_CREATE_VIRTUAL_PER:
                        sendUlt = copy.deepcopy(self.missing_ult)
                    else:
                        loseSerious = True


            # 发送补包完成信号：告知UI补包结束，是否丢包严重，以及需创建的虚假数据包
            self.pkg_doneSignal.emit(True, loseSerious, sendEmg, sendUlt)
            # 清空丢包列表（或许是重置接收编号列表）
            self._lostPackets['ult'].clear()
            self._lostPackets['emg'].clear()
            # 重置递归计数器
            self.current_retry = 0
            # 结束补包（停止定时器）
            self.finishSupplementPackage()
            return 0

        # 检查并截断超出配置最大包数的待补包列表
        self.checkMissingListIsOver()

        # 优先处理超声补包
        if len(self.missing_ult):
            # 首次处理时，记录最大超声补包数
            if self.m_ultSupPackCount == 0:
                self.m_maxUltPackCount = len(self.missing_ult)
            else:
                # 更新已补包数（总需补数 - 剩余待补数）
                self.m_ultSupPackCount = self.m_maxUltPackCount - len(self.missing_ult)
            # 发送补包总数信号（UI显示进度）
            self.pkgMaxCountSignal.emit("ult", self.m_maxUltPackCount)
            # 发送当前补包数信号（UI更新进度）
            self.pkgNowCountSignal.emit(self.m_ultSupPackCount)
            self.m_ultSupTimer.start()              # 启动超声补包定时器（触发批量补包）

        # 超声无待补包时，处理EMG补包
        elif len(self.missing_emg):
            self.startEmgSupPackage()

        # 无待补包时，递归调用自身（确认补包完成）
        else:
            self.supPackageRecursion()

    def startEmgSupPackage(self):
        """启动EMG补包流程（更新补包计数，触发定时器）"""
        # 首次处理时，记录最大EMG补包数
        if self.m_emgSupPackCount == 0:
            self.m_maxEmgPackCount = len(self.missing_emg)
        else:
            # 更新已补包数
            self.m_emgSupPackCount = self.m_maxEmgPackCount - len(self.missing_emg)
        # 发送补包总数/当前数信号（UI显示进度）
        self.pkgMaxCountSignal.emit("emg", self.m_maxEmgPackCount)
        self.pkgNowCountSignal.emit(self.m_emgSupPackCount)
        # 启动EMG补包定时器
        self.m_emgSupTimer.start()

    def ultrPackageTimeOutFunction(self):
        """超声补包定时器回调（批量请求补包）"""
        if self.m_supPackageMark==False:
            return self.supPackageRecursion()

        # print("ultr len:", len(self.missing_ult), self.m_ultSupPackCount)
        if len(self.missing_ult) > self.m_nowSupPackCount:
            supLen = 100
            surplus = len(self.missing_ult) - self.m_nowSupPackCount
            if surplus < 100:
                supLen = surplus

            supList = []
            for i in range(self.m_nowSupPackCount, self.m_nowSupPackCount+supLen):
                supList.append(self.missing_ult[i])
            # 批量请求补包（调用C# SDK）
            self.someRequestsPackage(supList, True)
            # self.requestPackage(self.missing_ult[self.m_nowSupPackCount], True)
            self.m_ultSupPackCount = self.m_ultSupPackCount + supLen
            self.m_nowSupPackCount = self.m_nowSupPackCount + supLen
            self.pkgNowCountSignal.emit(self.m_ultSupPackCount)
        else:
            self.m_ultSupTimer.stop()
            if len(self.missing_emg):
                self.m_nowSupPackCount = 0
                self.startEmgSupPackage()
            else:
                self.supPackageRecursion()

    def emgPackageTimeOutFunction(self):
        """EMG补包定时器回调（批量请求补包）"""
        # 强制中断补包时，终止流程并递归确认
        if self.m_supPackageMark==False:
            return self.supPackageRecursion()
        # print("emg len:", len(self.missing_emg), self.m_emgSupPackCount)
        # 仍有待补包时，批量请求补包（每次最多100个）
        if len(self.missing_emg) > self.m_nowSupPackCount:
            supLen = 100
            surplus = len(self.missing_emg) - self.m_nowSupPackCount
            if surplus < 100:
                supLen = surplus

            # 提取本次要补的包编号列表
            supList = []
            for i in range(self.m_nowSupPackCount, self.m_nowSupPackCount+supLen):
                supList.append(self.missing_emg[i])
            # 批量请求补包（调用C# SDK）
            self.someRequestsPackage(supList, False)  #是否超声信号
            # self.requestPackage(self.missing_emg[self.m_nowSupPackCount], False)
            # 更新补包计数
            self.m_nowSupPackCount = self.m_nowSupPackCount + supLen  #本轮已经补包的数量
            self.m_emgSupPackCount = self.m_emgSupPackCount + supLen  #已补包总数
            # 发送当前补包数信号（UI更新）
            self.pkgNowCountSignal.emit(self.m_emgSupPackCount)
        else:
            # EMG补包完成，停止定时器
            self.m_emgSupTimer.stop()
            # 递归确认补包完成
            self.supPackageRecursion()

    def supPackageRecursion(self):
        """补包递归入口（触发新一轮补包检查）"""
        # print(f"补包: {len(self.missing_ult)}  {len(self.missing_emg)}  title with test")
        # 终止条件：强制中断 或 无待补包
        if self.m_supPackageMark==False or (not self.missing_ult and not self.missing_emg):
            # 发送补包完成信号
            self.pkg_doneSignal.emit(True, False, None, None)
            # 清空丢包列表
            self._lostPackets['ult'].clear()
            self._lostPackets['emg'].clear()
            self.current_retry = 0  # 重置递归计数器
            # 结束补包
            self.finishSupplementPackage()
        else:
            # 递归调用补包逻辑
            self.find_discontinuities()

    def finishSupplementPackage(self):
        """结束补包流程（停止所有补包定时器）"""
        # 停止超声补包定时器
        if self.m_ultSupTimer.isActive():
            self.m_ultSupTimer.stop()
        # 停止EMG补包定时器
        if self.m_emgSupTimer.isActive():
            self.m_emgSupTimer.stop()


    def on_real_ultr_data_received(self, ultrasonicDataByChannel):
        return
        """
        处理来自不同通道的超声数据，并为每个数据数组发出一个信号。
        优化：减少不必要的列表转换，考虑批量处理信号发送。

        :param ultrasonicDataByChannel: 字典，键为通道标识符（int），
                                         值为整数数组的列表（List[int[]]）。
        """
        #print("ultr shishidedao: ", self._currUltPackNumber)
        # 遍历每个通道的超声数据
        print("hello2")
        for key, value_list in ultrasonicDataByChannel.items():
            for value in value_list:
                # 只在需要时进行类型转换
                # 类型转换：C#数组 → Python列表
                if not isinstance(value, list):
                    data_list = list(value)
                else:
                    data_list = value
                # 发送超声数据信号（供UI/保存模块处理）
                self.sendData_Sig.emit(key, data_list, False , self._currUltPackNumber)
                # 存储信号数据，等待批量发送  缓冲超声数据（用于LSL批量推送）
                self._ultrData.append(data_list)
        # 所有通道数据收集完成后，推送至LSL流
        if len(self._ultrData) == len(self._confUltChannels):
            # 转置数据（通道×采样点 → 采样点×通道）并推送
            self._ultrLSL.sendData(list(map(list, zip(*self._ultrData))))
            self._ultrData.clear()  #清空缓冲

    def update_plot(self):
        """后台线程：处理EMG数据缓冲，避免主线程阻塞"""
        while True:
            print("updata_plot")
            # 从队列取出待处理的EMG数据（阻塞直到有数据）
            butterData = self._data_queue.get() #放到队列里面，然后通过线程从队列里面取出来，防止在同一个线程函数下面，导致包编号和数据处理不过来
            # 遍历数据并发送信号（供UI绘图）
            print("updata—-plot")
            for itemG in butterData:
                 for key, value_list in itemG[1].items():
                     self.sendData_Sig.emit(key,value_list , True , itemG[0])
                 #print(itemG[0])
            # 标记队列任务完成
            self._data_queue.task_done()
    def on_real_emg_received(self, emgDataByChannel):
        """
        EMG数据接收事件回调（处理硬件推送的EMG数据）
        :param emgDataByChannel: 字典（key=通道号，value=float[] 数据数组）
        """
        #c# self._filter public float[] applyFiltering(float[] samples)
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        for key, value_list in emgDataByChannel.items():
            # # 新增：发射肌电专属信号（重点！）
            self.emgData_Sig.emit(key, list(value_list), self._currEmgPackNumber)
            self.sendData_Sig.emit(key, list(value_list), True, self._currEmgPackNumber)
            # for value in value_list:
            #     # 只在需要时进行类型转换
            #     # 类型转换：C#数组 → Python列表
            #     if not isinstance(value, list):
            #         data_list = list(value)
            #     else:
            #         data_list = value
            #     # 发送超声数据信号（供UI/保存模块处理）
            #     self.sendData_Sig.emit(key, data_list, True , self._currEmgPackNumber)
        #print(f"EMG Data received: {emgDataByChannel}")
        #print("emg shishidedao: " , self._currEmgPackNumber) #调试预留备注
        # 历史逻辑：旧版正常录制/补包的数据缓冲排序（已移至data_process函数，保留仅作参考）
        # if self.isReplenishedPackage: #录制结束，开始补包为True
        #     for key, value_list in emgDataByChannel.items():
        #         self.sendData_Sig.emit(key, list(value_list), True, self._currEmgPackNumber)
        # else: #正常录制和正常显示是在这里面执行
        #     if self._maxEmgPackNumber < 0:
        #         return
        #     if len(self._waveBuffer) >= 10: #缓冲10帧数据，然后进行排序，假如10帧之后包顺序还有乱的，就继续等待
        #         self._waveBuffer.sort(key=lambda x: x[0])
        #         self._maxEmgPackNumber = max(self._maxEmgPackNumber , self._waveBuffer[len(self._waveBuffer) - 1][0])
        #         if self._currEmgPackNumber > self._maxEmgPackNumber:
        #             self._data_queue.put(copy.deepcopy(self._waveBuffer))
        #             self._waveBuffer.clear()
        #     if self._currEmgPackNumber > self._maxEmgPackNumber:
        #         self._waveBuffer.append((self._currEmgPackNumber , {k: list(v) for k, v in emgDataByChannel.items()}))

    def trygogoo(self, emgDataByChannel):
        print("trygogoog")
    def cleanWaveBuffer(self):
        """清空EMG数据缓冲（采集停止时调用）"""
        # 重置最大包编号（停止缓冲排序）拒绝接收停止后的任何无效数据。
        self._maxEmgPackNumber = -1
        # 清空缓冲列表
        self._waveBuffer.clear()
        self._waveBuffer = []
        time.sleep(2) #忽视停止采集之后数据
        self._maxEmgPackNumber = 0  #把最大包编号重置为初始值 0，为下一次采集做准备


    def on_real_emg_shedding_received(self, shedding):
        #print(list(shedding), self.warning)
        # if self.warning:
        #     # 目前处于弹窗状态,信号丢弃
        #     return 0

        # self.setWarning(True)
        """
        EMG丢包事件回调（处理硬件推送的丢包通知）
        :param shedding: C#数组（丢包包编号）
        """
        print("on_real_emg_shedding_received")
        # 转换为Python列表
        sheddList = list(shedding)
        # 无丢包时直接返回
        if len(sheddList) == 0:
            # self.setWarning(False)
            return 0
        # 发送EMG丢包信号（供UI提示）
        self.emgSheddSignal.emit(sheddList)

    #Dictionary<int (通道), Dictionary<int (IMUDataType), List<short>  (IMUDataType对应后面的值)>
    def on_real_imu_received(self, imuByData):
        """
        IMU数据接收事件回调（处理硬件推送的IMU数据）
        :param imuByData: 嵌套字典（key=通道号，value=字典（key=IMUDataType，value=List<short> 数据））
        """
        print("IMU")
        # 存储所有IMU数据（供UI处理）
        imuAllData = {}
        # 存储扁平化的IMU数据（供LSL推送）
        imuSumData = []
        # 遍历每个通道的IMU数据
        for key, value_list in imuByData.items():
            imuData = {}
            # 遍历IMU数据类型（如加速度、角速度、角度）
            for keySub , value in value_list.items():
                pyData = [item for item in value]
                imuData[keySub] = pyData
                imuSumData.append(imuData)
            imuAllData[key] = imuData
         # 发送IMU数据信号（供UI/保存模块处理）
        self.IMUData_send_Sig.emit(imuAllData)
        # 推送IMU数据至LSL流（转置后推送）
        self._imuLSL.sendData(list(map(list, zip(*imuSumData))))

    #下面是下发命令到设备上得,  ultr:"" 有几个通道就写几个下标
    def configParam(self, ultr, emg, imu, inputMod, outMod, emgMod):
        """
        下发设备配置参数
        :param ultr: 超声配置
        :param emg: EMG配置
        :param imu: IMU配置
        :param inputMod: 输入模式
        :param outMod: 输出模式
        :param emgMod: EMG模式
        """
        self._newsletter.configParam(ultr, emg, imu, inputMod, outMod, emgMod,20)

    def deviceSwitch(self, isConnect):
        """
        设备连接开关
        :param isConnect: True=连接，False=断开
        """
        self._newsletter.deviceSwitch(isConnect)
        print("连接成功")

    def collectionSwitch(self, isCollection):
        """
         采集启停开关
         :param isCollection: True=启动采集，False=停止采集
         """
        self._newsletter.collectionSwitch(isCollection)
        self.cleanWaveBuffer()

    def triggerSignal(self,value):
        """
        发送触发信号
        :param value: 触发值
        """
        self._newsletter.triggerSignal(value)
    def triggerRecordSignal(self,isRecord):
        """
        发送录制触发信号
        :param isRecord: True=开始录制，False=停止录制
        """
        value = 0
        if isRecord:
            value = 1
        else:
            value = 0
        self._newsletter.triggerRecordSignal(value.to_bytes(1, byteorder='big'))
    def requestPackage(self , packNumber , isUlt):
        """
        单包补包请求
        :param packNumber: 包编号
        :param isUlt: True=超声包，False=EMG包
        """
        self._newsletter.requestPackage(packNumber , isUlt)

    def someRequestsPackage(self, packNumbersList, isUlt):
        """
        批量补包请求
        :param packNumbersList: 包编号列表
        :param isUlt: True=超声包，False=EMG包
        """
        self._newsletter.someRequestsPackage(packNumbersList, isUlt)

#------------------------------------3.设备搜索类-------------------------------------------
class DeviceSearcher(QObject):
    """基于mDNS的局域网设备搜索类（继承QObject以支持Qt信号）"""
    # Qt信号：发送搜索到的设备IP列表
    sendSearcherSig = pyqtSignal(list)
    class MyListener(ServiceListener):
        """mDNS服务监听器（内部类）"""
        def __init__(self, outer_instance):
            # 外部类实例引用（用于发送Qt信号）
            self.outer_instance = outer_instance
            super().__init__()
        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """服务更新回调"""
            print(f"Service {name} updated")

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """服务移除回调"""
            print(f"Service {name} removed")

        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            """服务发现回调（找到设备时触发）"""
            # 获取服务信息（包含设备IP）
            info = zc.get_service_info(type_, name)
            print(info)
            # 发送设备IP列表信号（供UI处理）
            self.outer_instance.sendSearcherSig.emit(info.parsed_addresses())
            #print(info.parsed_addresses()) #结果是list类型
    def __init__(self):
        """初始化设备搜索器"""
        super(DeviceSearcher, self).__init__()
        # 创建Zeroconf实例（mDNS核心）
        zeroconf = Zeroconf()
        # 创建服务监听器（关联外部类实例）
        listener = self.MyListener(self)
        # 启动服务浏览器：搜索_http._udp.local.类型的服务（设备广播的服务类型）
        ServiceBrowser(zeroconf, "_http._udp.local.", listener)

if __name__ == "__main__":
    app = QApplication([])
    print("程序启动成功")
    app.exec_()

"""
*bug函数:on_real_RelData_Received()，find_discontinuities()
*bug描述：偶发性出现递归循环。

*分析：
发生递归死循环的原因可能有多个，特别是在数据包处理和网络通讯的上下文中。下面是一些可能导致你的代码出现偶发性递归死循环的原因：

重复请求相同的数据包：如果requestPackage函数无法成功获取或重新传输丢失的数据包，那么丢失的数据包列表将不会清空，导致find_discontinuities函数不断被重新调用。

处理状态未正确更新：在递归调用find_discontinuities之前，如果丢失的数据包列表没有被正确更新或清空，函数可能会反复处理相同的数据包信息。

网络延迟或数据包传输失败：网络问题或数据包传输失败可能导致请求的数据包没有及时到达，使得程序反复尝试处理相同的丢包情况。

同步问题：如果多个线程或进程访问和修改共享的数据包记录（如_lostPackets），可能导致数据不一致，从而反复触发递归调用。

测试策略
为了测试并定位递归死循环的具体原因，你可以加入一些日志记录和条件断点：

添加日志记录：在递归函数的关键位置（如调用requestPackage和重新调用find_discontinuities之前）添加日志记录，记录关键变量的状态和函数调用的次数。

设置最大递归深度：可以在find_discontinuities函数中设置一个递归深度计数器，当递归调用达到某个阈值时停止递归，这有助于防止程序无限递归。

测试网络和请求的稳定性：确保requestPackage函数能够正确处理网络延迟和错误，且能够有效地从失败中恢复。

使用断点调试：在IDE中设置条件断点，比如当特定数据包号重复出现时触发，以此来调试和观察程序的行为。

下一步再遇见这种情况该怎么办和重大怀疑：
再次遇见死循环，就马上打开Wireshark抓包工具，看下发送请求之后下位机有没有回包，假如请求补包，没有回包就会发送死循环递归了，怀疑主要是这个原因。
假如不是话，那么在关键位置打印日志，看下问题所在。
"""