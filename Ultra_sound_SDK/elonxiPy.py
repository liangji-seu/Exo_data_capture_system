from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from pythonnet import load
from PyQt5.QtWidgets import QApplication
load("coreclr")
import queue
import clr
import copy
from scipy.signal import butter, filtfilt
import asyncio
import numpy as np
import time
import threading
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
import socket
clr.AddReference('System.Collections')
clr.AddReference("./无线/Elonxi_SDK")

from Elonxi_SDK import Newsletter,GlobalEvents,PacketType
from System.Collections.Generic import List

from scipy.signal import butter, filtfilt
from logic.LSLProtocol import SendSLStream
import copy
class EMGFilter:
    """
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
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandstop(self, lowcut, highcut, order=2):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a

    def filter_signal(self, data):
        #print(len(data))
        data = filtfilt(self.b_bandpass, self.a_bandpass, data,padlen=len(data)-1)
        data = filtfilt(self.b_bandstop, self.a_bandstop, data,padlen=len(data)-1)
        return data

    def filter_signal_offline(self, data, harmonics):
        b, a = self.butter_bandstop(harmonics - 2, harmonics + 2, order=2)
        data = filtfilt(self.b_bandpass, self.a_bandpass, data)
        data = filtfilt(b, a, data)
        return data


class NewsletterBase(QObject):
    receiveSignal = pyqtSignal(object, object)
    emgSheddSignal = pyqtSignal(object)
    sendData_Sig = pyqtSignal(int, list , bool , int) #bool参数是否emg数据
    IMUData_send_Sig = pyqtSignal(dict)
    pkg_doneSignal = pyqtSignal(bool, object, object, object)
    pkgMaxCountSignal = pyqtSignal(object, object)
    pkgNowCountSignal = pyqtSignal(object)
    def __init__(self,localPort,remoteAddress,remotePort):
        super(NewsletterBase, self).__init__()
        # self._filter = FilterProcessor()
        print(remoteAddress)
        self.MAX_CREATE_VIRTUAL_PER = 0.05 #创建虚假数据的最大比例,丢包数大于这个比例,则不创建虚假数据

        self._newsletter = Newsletter(localPort,remoteAddress,remotePort)
        self.sheddDict = {}
        self._waveBuffer = []
        self.isReplenishedPackage = False #录制结束，开始补包为True
        self._currUltPackNumber = 0
        self._currEmgPackNumber = 0
        self._maxEmgPackNumber = 0
        self._lostPackets = {'ult':[],'emg':[]}
        # self.warning = False
        self._data_queue = queue.Queue()
        self._isStartSaveFile = False
        self.max_retries = 10  # 限制最大递归次数
        self.current_retry = 0  # 当前递归深度
        self.initSheddDict()
        self.initLSL()

        self.missing_ult = []
        self.missing_emg = []
        self.m_supPackageMark = True    # 强行中断的标志位, False 时强制终端循环
        # self.m_alreadySupPack = 0              # 第二次进入递归开始, 记录的已经补包的总包数
        self.m_emgSupPackCount = 0
        self.m_ultSupPackCount = 0
        self.m_nowSupPackCount = 0
        self.m_maxEmgPackCount = 0
        self.m_maxUltPackCount = 0
        self.m_supRare = 10

        self.m_lastEmgSupCount = 0
        self.m_lastUltSupCount = 0

        self.m_ConfigMaxPackCount = {} #从配置中获取到的, 包最多能有多少个

        self.m_emgSupTimer = QTimer()
        self.m_emgSupTimer.setInterval(1000)
        self.m_emgSupTimer.timeout.connect(self.emgPackageTimeOutFunction)

        self.m_ultSupTimer = QTimer()
        self.m_ultSupTimer.setInterval(1000)
        self.m_ultSupTimer.timeout.connect(self.ultrPackageTimeOutFunction)

        # 订阅GlobalEvents的事件
        GlobalEvents.NotificationReceived += self.on_notification_received
        GlobalEvents.RealRealUltrDataReceived += self.on_real_ultr_data_received
        GlobalEvents.RealRealRelDataReceived += self.on_real_RelData_Received
        GlobalEvents.RealRealEMGReceived += self.on_real_emg_received
        GlobalEvents.RealRealEmgSheddingReceived += self.on_real_emg_shedding_received
        GlobalEvents.RealReaIMUReceived += self.on_real_imu_received

        threading.Thread(target=self.update_plot, daemon=True).start()

    def initLSL(self):
        self._emgLSL = SendSLStream('emg',8,100)
        self._imuLSL = SendSLStream('imu', 36, 100)

    def setConfUltChannels(self,confUltChannels):
        self._confUltChannels = confUltChannels
        self._ultrData = []
        self._ultrLSL = SendSLStream('ultr', len(confUltChannels), 1000)

    def initSheddDict(self):
        self.sheddDict.clear()
        self.sheddDict["emg"] = []

    # def setWarning(self, state):
    #     self.warning = state

       # 事件处理函数
    def on_notification_received(self, onType, message):
        if(onType == PacketType.DeviceConnection):
            # 连接成功
            self.initSheddDict()

            print(f"deviceConnect info:{message}")
        elif(onType == PacketType.Configuration):
            # 下发成功
            print(f"config param info:{message}")
        elif(onType == PacketType.CollectionStatus):
            # 采集状态
            print(f"collection status info:{message}")
        elif(onType == PacketType.IsPCStopMark):
            #设备由谁停止, True 是上位机停止, False 是下位机停止
            print(f"stop mark:{message}")
        elif(onType == PacketType.BatteryCapacity):
            # 电池容量
            print(f"battery capecityLa:{message}")
        elif(onType == PacketType.IsDeviceOnline):
            # 设备是否在线
            print(f"Is device online:{message}")
        elif(onType == PacketType.TriggerTimestamp):
            print(f"Trigger Timestamp la:{message}")

        self.receiveSignal.emit(onType, message)

    def on_real_RelData_Received(self , isUlt , packNumber):
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
        pass
    def setStartSaveFile(self,state):
        self._isStartSaveFile = state
    def getStartSaveFile(self):
        return self._isStartSaveFile
    
    def setSupPackageMark(self, mark):
        self.m_supPackageMark = mark
        self.m_nowSupPackCount = 0
        self.m_emgSupPackCount = 0
        self.m_ultSupPackCount = 0

        self.m_lastEmgSupCount = 0
        self.m_lastUltSupCount = 0

    def initMaxPackCount(self, countCfg):
        self.m_ConfigMaxPackCount = countCfg

    def checkMissingListIsOver(self):
        if "ult" in self.m_ConfigMaxPackCount.keys():
            if len(self.missing_ult) > self.m_ConfigMaxPackCount["ult"]:
                if self.m_ultSupPackCount < self.m_ConfigMaxPackCount["ult"]:
                    self.missing_ult = self.missing_ult[self.m_ultSupPackCount:self.m_ConfigMaxPackCount["ult"]]
                else:
                    self.missing_ult = self.missing_ult[0:self.m_ConfigMaxPackCount["ult"]]

        if "emg" in self.m_ConfigMaxPackCount.keys():
            if len(self.missing_emg) > self.m_ConfigMaxPackCount["emg"]:
                if self.m_emgSupPackCount < self.m_ConfigMaxPackCount["emg"]:
                    self.missing_emg = self.missing_emg[self.m_emgSupPackCount:self.m_ConfigMaxPackCount["emg"]]
                else:
                    self.missing_emg = self.missing_emg[0:self.m_ConfigMaxPackCount["emg"]]

    def find_discontinuities(self):
        def sub(nums):
            nums = sorted(set(nums)) #去重和排序
            missing = []
            for i in range(len(nums) - 1):
                if nums[i + 1] - nums[i] > 1:
                    missing.extend(range(nums[i] + 1, nums[i + 1]))
            return missing
        
        self.m_nowSupPackCount = 0
        self.current_retry += 1
        #开始做补包处理
        #self.collectionSwitch(False)
        time.sleep(0.1)
        #QApplication.processEvents()
        ult = copy.deepcopy(self._lostPackets['ult'])
        emg = copy.deepcopy(self._lostPackets['emg'])
        self.missing_ult = sub(ult)
        self.missing_emg = sub(emg)

        ultStopMark = False
        emgStopMark = False

        if self.m_lastUltSupCount != len(self.missing_ult):
            self.m_lastUltSupCount = len(self.missing_ult)
        else:
            ultStopMark = True

        if self.m_lastEmgSupCount != len(self.missing_emg):
            self.m_lastEmgSupCount = len(self.missing_emg)
        else:
            emgStopMark = True

        if (ultStopMark and emgStopMark) or self.m_supPackageMark==False or (self.max_retries < self.current_retry):
            loseSerious = False
            sendEmg = []
            sendUlt = []

            # self.m_supPackageMark 为False 是停止补包,因此不进行数据造假
            if self.m_supPackageMark:
                emgMaxCount = len(self._lostPackets['emg'])
                ultMaxCount = len(self._lostPackets['ult'])
                if emgMaxCount != 0:
                    if len(self.missing_emg)/emgMaxCount <= self.MAX_CREATE_VIRTUAL_PER:
                        sendEmg = copy.deepcopy(self.missing_emg)
                    else:
                        loseSerious = True

                if ultMaxCount != 0:
                    if len(self.missing_ult)/ultMaxCount <= self.MAX_CREATE_VIRTUAL_PER:
                        sendUlt = copy.deepcopy(self.missing_ult)
                    else:
                        loseSerious = True

            self.pkg_doneSignal.emit(True, loseSerious, sendEmg, sendUlt)
            self._lostPackets['ult'].clear()
            self._lostPackets['emg'].clear()
            self.current_retry = 0  # 重置递归计数器
            self.finishSupplementPackage()
            return 0

        self.checkMissingListIsOver()

        if len(self.missing_ult):
            if self.m_ultSupPackCount == 0:
                self.m_maxUltPackCount = len(self.missing_ult)
            else:
                self.m_ultSupPackCount = self.m_maxUltPackCount - len(self.missing_ult)
            self.pkgMaxCountSignal.emit("ult", self.m_maxUltPackCount)  
            self.pkgNowCountSignal.emit(self.m_ultSupPackCount)
            self.m_ultSupTimer.start()

        elif len(self.missing_emg):
            self.startEmgSupPackage()

        else:
            self.supPackageRecursion()

    def startEmgSupPackage(self):
        if self.m_emgSupPackCount == 0:
            self.m_maxEmgPackCount = len(self.missing_emg)
        else:
            self.m_emgSupPackCount = self.m_maxEmgPackCount - len(self.missing_emg)
        self.pkgMaxCountSignal.emit("emg", self.m_maxEmgPackCount)
        self.pkgNowCountSignal.emit(self.m_emgSupPackCount)
        self.m_emgSupTimer.start()

    def ultrPackageTimeOutFunction(self):
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
        if self.m_supPackageMark==False:
            return self.supPackageRecursion()
        # print("emg len:", len(self.missing_emg), self.m_emgSupPackCount)
        if len(self.missing_emg) > self.m_nowSupPackCount:
            supLen = 100
            surplus = len(self.missing_emg) - self.m_nowSupPackCount
            if surplus < 100:
                supLen = surplus
            
            supList = []
            for i in range(self.m_nowSupPackCount, self.m_nowSupPackCount+supLen):
                supList.append(self.missing_emg[i])
            self.someRequestsPackage(supList, False)
            # self.requestPackage(self.missing_emg[self.m_nowSupPackCount], False)
            self.m_nowSupPackCount = self.m_nowSupPackCount + supLen
            self.m_emgSupPackCount = self.m_emgSupPackCount + supLen
            self.pkgNowCountSignal.emit(self.m_emgSupPackCount)
        else:
            self.m_emgSupTimer.stop()
            self.supPackageRecursion()

    def supPackageRecursion(self):
        # print(f"补包: {len(self.missing_ult)}  {len(self.missing_emg)}  title with test")
        if self.m_supPackageMark==False or (not self.missing_ult and not self.missing_emg):
            self.pkg_doneSignal.emit(True, False, None, None)
            self._lostPackets['ult'].clear()
            self._lostPackets['emg'].clear()
            self.current_retry = 0  # 重置递归计数器
            self.finishSupplementPackage()
        else:
            self.find_discontinuities()

    def finishSupplementPackage(self):
        if self.m_ultSupTimer.isActive():
            self.m_ultSupTimer.stop()

        if self.m_emgSupTimer.isActive():
            self.m_emgSupTimer.stop()


    def on_real_ultr_data_received(self, ultrasonicDataByChannel):
        """
        处理来自不同通道的超声数据，并为每个数据数组发出一个信号。
        优化：减少不必要的列表转换，考虑批量处理信号发送。

        :param ultrasonicDataByChannel: 字典，键为通道标识符（int），
                                         值为整数数组的列表（List[int[]]）。
        """
        #print("ultr shishidedao: ", self._currUltPackNumber)
        for key, value_list in ultrasonicDataByChannel.items():
            for value in value_list:
                # 只在需要时进行类型转换
                if not isinstance(value, list):
                    data_list = list(value)
                else:
                    data_list = value

                self.sendData_Sig.emit(key, data_list, False , self._currUltPackNumber)
                # 存储信号数据，等待批量发送
                self._ultrData.append(data_list)

        if len(self._ultrData) == len(self._confUltChannels):
            self._ultrLSL.sendData(list(map(list, zip(*self._ultrData))))
            self._ultrData.clear()

    def update_plot(self):
        while True:
            butterData = self._data_queue.get() #放到队列里面，然后通过线程从队列里面取出来，防止在同一个线程函数下面，导致包编号和数据处理不过来
            for itemG in butterData:
                 for key, value_list in itemG[1].items():
                     self.sendData_Sig.emit(key,value_list , True , itemG[0])
                 #print(itemG[0])
            self._data_queue.task_done()
    def on_real_emg_received(self, emgDataByChannel):
        #c# self._filter public float[] applyFiltering(float[] samples)
        for key, value_list in emgDataByChannel.items():
            self.sendData_Sig.emit(key, list(value_list), True, self._currEmgPackNumber)
        #print(f"EMG Data received: {emgDataByChannel}")
        #print("emg shishidedao: " , self._currEmgPackNumber)
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
    def cleanWaveBuffer(self):
        self._maxEmgPackNumber = -1
        self._waveBuffer.clear()
        self._waveBuffer = []
        time.sleep(2) #忽视停止采集之后数据
        self._maxEmgPackNumber = 0


    def on_real_emg_shedding_received(self, shedding):
        #print(list(shedding), self.warning)
        # if self.warning:
        #     # 目前处于弹窗状态,信号丢弃
        #     return 0

        # self.setWarning(True)

        sheddList = list(shedding)
        if len(sheddList) == 0:
            # self.setWarning(False)
            return 0
        
        self.emgSheddSignal.emit(sheddList)

    #Dictionary<int (通道), Dictionary<int (IMUDataType), List<short>  (IMUDataType对应后面的值)>
    def on_real_imu_received(self, imuByData):
        imuAllData = {}
        imuSumData = []
        for key, value_list in imuByData.items():
            imuData = {}
            for keySub , value in value_list.items():
                pyData = [item for item in value]
                imuData[keySub] = pyData
                imuSumData.append(imuData)
            imuAllData[key] = imuData

        self.IMUData_send_Sig.emit(imuAllData)
        self._imuLSL.sendData(list(map(list, zip(*imuSumData))))

    #下面是下发命令到设备上得,  ultr:"" 有几个通道就写几个下标
    def configParam(self, ultr, emg, imu, inputMod, outMod, emgMod):
        self._newsletter.configParam(ultr, emg, imu, inputMod, outMod, emgMod)

    def deviceSwitch(self, isConnect):
        self._newsletter.deviceSwitch(isConnect)

    def collectionSwitch(self, isCollection):
        self._newsletter.collectionSwitch(isCollection)
        self.cleanWaveBuffer()

    def triggerSignal(self,value):
        self._newsletter.triggerSignal(value)
    def triggerRecordSignal(self,isRecord):
        value = 0
        if isRecord:
            value = 1
        else:
            value = 0
        self._newsletter.triggerRecordSignal(value.to_bytes(1, byteorder='big'))
    def requestPackage(self , packNumber , isUlt):
        self._newsletter.requestPackage(packNumber , isUlt)

    def someRequestsPackage(self, packNumbersList, isUlt):
        self._newsletter.someRequestsPackage(packNumbersList, isUlt)

class DeviceSearcher(QObject):
    sendSearcherSig = pyqtSignal(list)
    class MyListener(ServiceListener):
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance
            super().__init__()
        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            print(f"Service {name} updated")

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            print(f"Service {name} removed")

        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            print(info)
            self.outer_instance.sendSearcherSig.emit(info.parsed_addresses())
            #print(info.parsed_addresses()) #结果是list类型
    def __init__(self):
        super(DeviceSearcher, self).__init__()
        zeroconf = Zeroconf()
        listener = self.MyListener(self)
        ServiceBrowser(zeroconf, "_http._udp.local.", listener)


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