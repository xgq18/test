import sys
import pandas as pd
import numpy as np
from math import *
import time
import matplotlib.pyplot as plt
import pyqtgraph as pg
import cgitb
from PyQt5.Qt import *
from TmpData import *
from Mythreading import *
from pyqt5_graph import *


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import tab_ui_test

def sampling(i, rand, lmda, k_wbl):  # 抽取随机数函数
    Rf = rand[i]
    tf = (-log(1 - Rf)) ** (1 / k_wbl) / lmda
    return tf

class MainCode(QMainWindow, tab_ui_test.Ui_MainWindow):
    #下面四个list用于记录方案的选择
    pump_low_items = []
    pump_high_items = []
    press_low_items =[]
    press_high_items = []
    def __init__(self):
        QMainWindow.__init__(self)
        tab_ui_test.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.interface_plus()

        self.pushButton_10.clicked.connect(self.press_drawing1)
        self.pushButton_9.clicked.connect(self.pump_drawing1)
        self.pushButton_11.clicked.connect(self.pump_drawing2)
        self.pushButton_12.clicked.connect(self.press_drawing2)

    def pump_drawing1(self):
        self.verticalLayout_9.removeWidget(self.plt9)
        pg.setConfigOption("background", "w")
        self.plt9 = pg.PlotWidget()
        self.plt9.addLegend(size=(150, 80))
        self.plt9.showGrid(x=True, y=True, alpha=0.5)
        for pump_low_index in range(len(self.pump_low_items)):
            self.pump_low_items[pump_low_index] = int(self.pump_low_items[pump_low_index])
        for pump_high_index in range(len(self.pump_high_items)):
            self.pump_high_items[pump_high_index] = int(self.pump_high_items[pump_high_index])
        self.pump_high_items.sort()
        self.pump_low_items.sort()
        color = [(0,0,200),(0,128,0),(19,234,201),(195,46,212),(250,194,5),(54,55,55),(0,114,189),(217,83,25),(237,177,32),(126,47,142),(119,172,48),"r","c","k"]
        mn = 0
        cost_min_list = []
        cost_min_index_list = []
        df = pd.read_csv(r"D:\课题组任务\青岛LNG维修\11月模型\数据\cost_data1.csv",index_col=[0])
        self.tp_list = np.linspace(6000,14000,161)
        for m in self.pump_low_items:
            for n in self.pump_high_items:
                # self.pump_data()
                if m == 1 and n == 1:
                    # self.pump_count(1, 1)
                    self.cost11 = df.iloc[0, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案11")
                elif m == 1 and n == 2:
                    #2和5是有问题的
                    # self.pump_count(1, 2)
                    self.cost11 = df.iloc[1, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案12")

                elif m == 1 and n == 3:
                    # self.pump_count(1, 3)
                    self.cost11 = df.iloc[2, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案13")

                elif m == 1 and n == 4:
                    self.pump_count(1, 4)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案14")

                elif m == 1 and n == 5:
                    # self.pump_count(1, 5)
                    self.cost11 = df.iloc[3, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案15")

                elif m == 1 and n == 6:
                    # self.pump_count(1, 6)
                    self.cost11 = df.iloc[1, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案16")
                elif m == 1 and n == 7:
                    # self.pump_count(1, 7)
                    self.cost11 = df.iloc[2, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案17")
                elif m == 2 and n == 1:
                    self.pump_count(2, 1)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案21")
                elif m == 2 and n == 2:
                    self.pump_count(2, 2)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案22")
                elif m == 2 and n == 3:
                    self.pump_count(2, 3)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案23")
                elif m == 2 and n == 4:
                    self.pump_count(2, 4)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案24")
                elif m == 2 and n == 5:
                    self.pump_count(2, 5)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案25")
                elif m == 2 and n == 6:
                    self.pump_count(2, 6)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案26")
                elif m == 2 and n == 7:
                    self.pump_count(2, 7)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt9.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案27")
                mn += 1
        strategy_number = cost_min_list.index(min(cost_min_list))
        if len(self.pump_low_items) == 1:
            stratedy_low = self.pump_low_items[0]
            strategy_high = strategy_number
        else:
            stratedy_low = strategy_number // len(self.pump_high_items) +1
            strategy_high = strategy_number % len(self.pump_high_items)
        txt = "方案"+str(stratedy_low)+str(self.pump_high_items[strategy_high])
        tp = self.tp_list[cost_min_index_list[strategy_number]]
        self.lineEdit_7.setText(txt)
        self.lineEdit_8.setText(str(tp))
        self.verticalLayout_9.addWidget(self.plt9)

        for pump_low_index in range(len(self.pump_low_items)):
            self.pump_low_items[pump_low_index] = str(self.pump_low_items[pump_low_index])
        for pump_high_index in range(len(self.pump_high_items)):
            self.pump_high_items[pump_high_index] = str(self.pump_high_items[pump_high_index])

    def press_drawing1(self):
        self.verticalLayout_11.removeWidget(self.plt11)
        pg.setConfigOption("background", "w")
        self.plt11 = pg.PlotWidget()
        self.plt11.addLegend(size=(150, 80))
        self.plt11.showGrid(x=True, y=True, alpha=0.5)

        for press_low_index in range(len(self.press_low_items)):
            self.press_low_items[press_low_index] = int(self.press_low_items[press_low_index])
        for press_high_index in range(len(self.press_high_items)):
            self.press_high_items[press_high_index] = int(self.press_high_items[press_high_index])
        self.press_high_items.sort()
        self.press_low_items.sort()
        color = [(0,0,200),(0,128,0),(19,234,201),(195,46,212),(250,194,5),(54,55,55),(0,114,189),(217,83,25),(237,177,32),(126,47,142),(119,172,48),"r","c","k"]
        mn = 0
        cost_min_list = []
        cost_min_index_list = []
        df = pd.read_csv(r"D:\课题组任务\青岛LNG维修\11月模型\数据\压缩机低负荷数据.csv",index_col=[0])
        self.tp_list = np.linspace(6000,14000,161)
        for m in self.press_low_items:
            for n in self.press_high_items:
                # self.pump_data()
                if m == 1 and n == 1:
                    # self.pump_count(1, 1)
                    print("xxx")
                    self.cost11 = df.iloc[0, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt11.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案11")
                elif m == 1 and n == 2:
                    #2和5是有问题的
                    # self.pump_count(1, 2)
                    # self.cost11 = df.iloc[1, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt11.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案12")
                elif m == 2 and n == 1:
                    # self.pump_count(2, 1)
                    self.cost11 = df.iloc[1, :].tolist()
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt11.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案21")
                elif m == 2 and n == 2:
                    self.pump_count(2, 2)
                    cost_min_list.append(min(self.cost11))
                    cost_min_index_list.append(self.cost11.index(min(self.cost11)))
                    self.plt11.plot(x=self.tp_list, y=self.cost11, pen=color[mn], name="方案22")
                mn += 1
        strategy_number = cost_min_list.index(min(cost_min_list))
        if len(self.press_low_items) == 1:
            stratedy_low = self.press_low_items[0]
            strategy_high = strategy_number
        else:
            stratedy_low = strategy_number // len(self.press_high_items) +1
            strategy_high = strategy_number % len(self.press_high_items)
        txt = "方案"+str(stratedy_low)+str(self.press_high_items[strategy_high])
        tp = self.tp_list[cost_min_index_list[strategy_number]]
        self.lineEdit_9.setText(txt)
        self.lineEdit_10.setText(str(tp))
        self.verticalLayout_11.addWidget(self.plt11)

        for press_low_index in range(len(self.press_low_items)):
            self.press_low_items[press_low_index] = str(self.press_low_items[press_low_index])
        for press_high_index in range(len(self.press_high_items)):
            self.press_high_items[press_high_index] = str(self.press_high_items[press_high_index])

    def pump_drawing2(self):
        self.verticalLayout_13.removeWidget(self.plt13)
        pg.setConfigOption("background", "w")
        self.plt13 = pg.PlotWidget()
        self.plt13.addLegend(size=(150, 80))
        self.plt13.showGrid(x=True, y=True, alpha=0.5)
        df = pd.read_csv(r"D:\课题组任务\青岛LNG维修\11月模型\数据\高压泵策略对比.csv",index_col=[0])
        label_list = ["事后维修", "预防性维修","健康评估支持下的事后维修","健康评估支持下的预防性维修"]
        color = ["r", "c", "k","y"]
        cost_min_list = []
        cost_min_index_list = []
        self.tp_list = np.linspace(6000, 14000, 161)
        for mn in range(4):
            self.plt13.plot(x=self.tp_list,y=df.iloc[mn, :].tolist(),pen=color[mn],name=label_list[mn])
            self.cost11 = df.iloc[mn, :].tolist()
            cost_min_list.append(min(self.cost11))
            cost_min_index_list.append(self.cost11.index(min(self.cost11)))
            # self.plt13.plot(x=self.tp_list, y=[mn]*len(self.tp_list), pen=color[mn], name=label_list[mn])
        strategy_number = cost_min_list.index(min(cost_min_list))
        self.lineEdit_5.setText(label_list[strategy_number]  )
        self.verticalLayout_13.addWidget(self.plt13)

    def press_drawing2(self):
        self.verticalLayout_14.removeWidget(self.plt14)
        pg.setConfigOption("background", "w")
        self.plt14 = pg.PlotWidget()
        self.plt14.addLegend(size=(150, 80))
        self.plt14.showGrid(x=True, y=True, alpha=0.5)
        df = pd.read_csv(r"D:\课题组任务\青岛LNG维修\11月模型\数据\压缩机四种策略对比.csv",index_col=[0])
        label_list = ["事后维修", "预防性维修","健康评估支持下的事后维修","健康评估支持下的预防性维修"]
        color = ["r", "c", "k","y"]
        cost_min_list = []
        cost_min_index_list = []
        self.tp_list = np.linspace(6000, 14000, 161)
        for mn in range(4):
            self.plt14.plot(x=self.tp_list,y=df.iloc[mn, :].tolist(),pen=color[mn],name=label_list[mn])
            self.cost11 = df.iloc[mn, :].tolist()
            cost_min_list.append(min(self.cost11))
            cost_min_index_list.append(self.cost11.index(min(self.cost11)))
        strategy_number = cost_min_list.index(min(cost_min_list))
        self.lineEdit_6.setText(label_list[strategy_number]  )
        self.verticalLayout_14.addWidget(self.plt14)


    #界面增加画图的控件
    def interface_plus(self):
        pg.setConfigOption("background", "w")
        self.plt11 = pg.PlotWidget()
        self.plt11.addLegend(size=(150, 80))
        self.plt11.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout_11.addWidget(self.plt11)

        self.plt9 = pg.PlotWidget()
        self.plt9.addLegend(size=(150, 80))
        self.plt9.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout_9.addWidget(self.plt9)

        self.plt13 = pg.PlotWidget()
        self.plt13.addLegend(size=(150, 80))
        self.plt13.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout_13.addWidget(self.plt13)

        self.plt14 = pg.PlotWidget()
        self.plt14.addLegend(size=(150, 80))
        self.plt14.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout_14.addWidget(self.plt14)

        # 槽函数连接
        #界面补充
        # self.pushButton_10.clicked.connect(self.press11_count)
        # self.pushButton_9.clicked.connect(self.pump11_count)
        # self.pushButton_11.clicked.connect(self.pump_strategy)
        # self.pushButton_12.clicked.connect(self.press_strategy)
        #高压泵低负荷方案
        self.checkBox_14.stateChanged.connect(lambda: self.pump_low_cb(self.checkBox_14))
        self.checkBox_15.stateChanged.connect(lambda: self.pump_low_cb(self.checkBox_15))
        #高雅而并高负荷期方案
        self.checkBox.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox))
        self.checkBox_9.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox_9))
        self.checkBox_2.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox_2))
        self.checkBox_10.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox_10))
        self.checkBox_11.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox_11))
        self.checkBox_12.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox_12))
        self.checkBox_13.stateChanged.connect(lambda: self.pump_high_cb(self.checkBox_13))
        #压缩机高负荷期方案
        self.checkBox_3.stateChanged.connect(lambda: self.press_high_cb(self.checkBox_3))
        self.checkBox_16.stateChanged.connect(lambda: self.press_high_cb(self.checkBox_16))
        #压缩机低负荷方案
        self.checkBox_21.stateChanged.connect(lambda: self.press_low_cb(self.checkBox_21))
        self.checkBox_22.stateChanged.connect(lambda: self.press_low_cb(self.checkBox_22))

    def pump_count(self, m, n):
        self.pump_data()
        self.high_to_low_number = n
        self.low_change = m
        for tp_time in self.tp_list:
            self.tp = tp_time
            self.y1 = 0
            self.y2 = 0
            self.y4 = 0
            self.tf_low = 0
            self.tf_high = 0
            self.tp_count = 0
            self.t_low_overlap = []
            self.t_high_overlap = []
            for times in range(self.cycle_times):
                self.t_record = pd.DataFrame([0 for i in range(self.number_high)])
                s_record_temp = [1 for i in range(self.number_low)]
                for i in range(self.number_minus):
                    s_record_temp.append(3)
                self.s_record = pd.DataFrame(s_record_temp)
                self.t_fault_list = [self.life1[i + self.y1] for i in range(self.number_low)]
                self.y1 += self.number_low
                for i in range(self.number_minus):
                    self.t_fault_list.append((self.life2[self.y2]))
                    self.y2 += 1
                self.x = 0
                self.t_now_list = [0]
                self.operation = 2
                for m in range(len(self.t_table)):
                    self.m = m
                    self.th_start = self.t_table.iloc[m, 0]
                    index_using = [i for i in range(self.number_high) if self.s_record.iloc[i, -1] == 1]
                    t_record_using = self.t_record.iloc[index_using, -1]
                    t_fault_using = np.array(self.t_fault_list)[index_using]
                    dtf = (t_fault_using - t_record_using).tolist()
                    dtp = (self.tp - t_record_using).tolist()
                    dtf_min = min(dtf)
                    dtp_min = min(dtp)
                    self.t_next_fault = self.t_now_list[-1] + dtf_min
                    self.t_next_pm = self.t_now_list[-1] + dtp_min
                    if self.t_next_pm < self.t_next_fault:
                        self.operation = 1
                        self.t_next_fault = self.t_next_pm
                    else:
                        self.operation = 2
                    times_stop_count_temp = 0
                    t_low_stop_temp = []
                    th_start_temp = self.th_start - 240 * self.number_high
                    while self.t_next_fault < th_start_temp:
                        times_stop_count_temp += 1
                        t_low_stop_temp.append(self.t_next_fault)
                        if self.operation == 2:
                            self.tf_low += 1
                            self.pump_fault_in_low()
                        else:
                            self.tp_count += 1
                            self.pump_pm_in_low()

                        index_using = [i for i in range(self.number_high) if self.s_record.iloc[i, -1] == 1]
                        t_record_using = self.t_record.iloc[index_using, -1]
                        t_fault_using = np.array(self.t_fault_list)[index_using]
                        dtf = (t_fault_using - t_record_using).tolist()
                        dtp = (self.tp - t_record_using).tolist()
                        dtf_min = min(dtf)
                        dtp_min = min(dtp)
                        self.t_next_fault = self.t_now_list[-1] + dtf_min
                        self.t_next_pm = self.t_now_list[-1] + dtp_min
                        if self.t_next_pm < self.t_next_fault:
                            self.operation = 1
                            self.t_next_fault = self.t_next_pm
                        else:
                            self.operation = 2

                    if times_stop_count_temp >= self.number_minus + 1:
                        for index_temp in range(times_stop_count_temp - self.number_minus):
                            rand_temp = self.rand1[self.y4]
                            self.y4 += 1
                            if rand_temp > self.rate_self_maintenance:
                                maintenance_t = self.stop_day_out * 24
                            else:
                                maintenance_t = self.stop_day_self * 24
                            if t_low_stop_temp[index_temp + self.number_minus] <= t_low_stop_temp[
                                index_temp] + maintenance_t:
                                t_overlap = t_low_stop_temp[index_temp] + maintenance_t - t_low_stop_temp[
                                    index_temp + self.number_minus]
                                self.t_low_overlap.append(t_overlap)

                    self.th_end = self.t_table.iloc[m, 1]
                    th_start_list = [th_start_temp + i * 240 for i in range(self.number_high)]
                    for t_now in th_start_list:
                        index_using = [i for i in range(self.number_high) if self.s_record.iloc[i, -1] == 1]
                        t_record_temp = self.t_record.iloc[index_using, -1].tolist()
                        index_pm = index_using[t_record_temp.index(max(t_record_temp))]
                        dt_plus = t_now - self.t_now_list[-1]
                        t_pm_now = self.t_record.iloc[index_pm, -1] + dt_plus
                        if t_pm_now + (self.th_end - t_now) + self.th_time >= self.tp:
                            self.tp_count += 1
                            s_record_temp = self.s_record.iloc[:, -1]
                            index_spare = [i for i in range(self.number_high) if self.s_record.iloc[i, -1] != 1]
                            t_operation_spare = self.t_record.iloc[index_spare, -1].tolist()
                            index_change = index_spare[t_operation_spare.index(min(t_operation_spare))]
                            if index_pm in self.pump_life2:
                                self.t_fault_list[index_pm] = self.life2[self.y2]
                                self.y2 += 1
                            else:
                                self.t_fault_list[index_pm] = self.life1[self.y1]
                                self.y1 += 1
                            s_record_temp[index_change] = 1
                            s_record_temp[index_pm] = 4
                            self.s_record[self.x + 1] = s_record_temp
                            t_record_temp = self.t_record[self.x].tolist()
                            for i in index_using:
                                t_record_temp[i] += dt_plus
                            t_record_temp[index_pm] = 0
                            self.t_record[self.x + 1] = t_record_temp
                            self.x += 1
                            self.t_now_list.append(t_now)

                    t_record_temp = self.t_record.iloc[:, -1]
                    index_tf = [i for i in range(self.number_high) if self.t_fault_list[i] - t_record_temp[i] <= 0]
                    if len(index_tf) > 0:
                        self.tf_low += len(index_tf)
                        for index_tflow in index_tf:
                            self.t_record.iloc[index_tflow, -1] = 0
                            if index_tflow in self.pump_life2:
                                self.t_fault_list[index_tflow] = self.life2[self.y2]
                                self.y2 += 1
                            else:
                                self.t_fault_list[index_tflow] = self.life1[self.y1]
                                self.y1 += 1

                    self.pump_low_to_high()
                    dtf_high = [self.t_fault_list[i] - self.t_record.iloc[i, -1] for i in range(self.number_high)]
                    dtf_high_min = min(dtf_high)
                    self.t_next_fault = self.t_now_list[-1] + dtf_high_min
                    tf_high_temp = 0
                    while self.t_next_fault < self.th_end:
                        tf_high_temp += 1
                        self.tf_high += 1
                        self.pump_fault_in_high()

                    self.t_high_overlap.append(tf_high_temp * 360)
                    self.pump_high_to_low()
            self.tf_high_list.append(self.tf_high / self.cycle_times)
            self.tf_low_list.append(self.tf_low / self.cycle_times)
            self.tp_count_list.append(self.tp_count / self.cycle_times)
            t_stop = sum(self.t_low_overlap) / self.cycle_times / 4
            self.t_low_overlap_list.append(t_stop * 4 + 0.000001)
            t_high_stop = sum(self.t_high_overlap) / self.cycle_times / 6
            self.Avai_low.append((5880 * 25 - t_stop) / 5880 / 25)
            self.Avai_high.append((2880 * 25 - t_high_stop) / 2880 / 25)


        cost_list = []
        self.cost_rand = self.rand1
        for i in range(len(self.tp_list)):
            tf_high_times = int(self.tf_high_list[i] * self.cycle_times)
            tf_low_times = int(self.tf_low_list[i] * self.cycle_times)
            tp_times = int(self.tp_count_list[i] * self.cycle_times)
            hrs = self.t_low_overlap_list[i] * self.cycle_times
            e1 = self.pump_cost_high_fault(tf_high_times)
            e = e1 + self.pump_cost_low_fault(tf_low_times) + self.pump_cost_low_pm(tp_times) + self.pump_cost_downtime(hrs)
            cost_list.append(e/self.cycle_times)
        self.cost11 = cost_list
        # self.df = pd.DataFrame([self.tf_high_list,self.tf_low_list,self.tp_count_list,self.Avai_low,self.Avai_high,self.tp_list,self.cost11],
        #           index=["tf_high","tf_low","tp_count","低负荷可用度", "高负荷期可用度","预防性维修限值","费用"])

    def press11_count(self):
        self.press_data()
        for tp_time in self.tp_list:
            self.tp = tp_time
            self.y1 = 0
            self.y2 = 0
            self.tf_low_homemade = 0
            self.tp_homemade = 0
            self.tp_imported = 0
            self.tf_high_imported = 0
            self.tf_high_homemade = 0
            for times in range(self.cycle_times):
                self.t_record = pd.DataFrame([0, 0, 0])
                self.s_record = pd.DataFrame([1, 3, 3])
                self.t_fault_list = [self.life1[self.y1]]
                self.y1 += 1
                for i in range(2):
                    self.t_fault_list.append(self.life2[i + self.y2])
                self.y2 += 2
                self.x = 0
                self.t_now_list = [0]
                for m in range(len(self.t_table)):
                    th_start = self.t_table.iloc[m, 0]
                    self.th_start = th_start
                    flag_temp = self.t_record.iloc[0, -1] // 3000 + 1
                    dtp_small = flag_temp * 3000 - self.t_record.iloc[0, -1]
                    dtf = self.t_fault_list[0] - self.t_record.iloc[0, -1]
                    dtp_large = self.tp - self.t_record.iloc[0, -1]
                    dt_min = min(dtp_small, dtp_large, dtf)
                    self.t_next_fault = self.t_now_list[-1] + dt_min
                    th_start_temp = th_start - self.operation_number * self.homemade_time
                    if dt_min == dtp_large:
                        flag = 1
                    elif dt_min == dtp_small:
                        flag = 2
                    else:
                        flag = 3

                    while self.t_next_fault < th_start_temp:
                        if flag == 1:
                            self.tp_homemade += 1
                            self.press_pm_in_low11()
                        elif flag == 2:
                            self.press_pm_small()
                        else:
                            self.tf_low_homemade += 1
                            self.press_fault_in_low11()
                        flag_temp = self.t_record.iloc[0, -1] // 3000 + 1
                        dtp_small = flag_temp * 3000 - self.t_record.iloc[0, -1]
                        dtf = self.t_fault_list[0] - self.t_record.iloc[0, -1]
                        dtp_large = self.tp - self.t_record.iloc[0, -1]
                        dt_min = min(dtp_small, dtp_large, dtf)
                        self.t_next_fault = self.t_now_list[-1] + dt_min
                        th_start_temp = th_start - self.operation_number* self.homemade_time
                        if dt_min == dtp_large:
                            flag = 1
                        elif dt_min == dtp_small:
                            flag = 2
                        else:
                            flag = 3

                    th_end = self.t_table.iloc[m, 1]
                    self.th_end = th_end
                    for i_pm in range(self.operation_number ):
                        i_pm = 2 - i_pm
                        if i_pm != 0:
                            t_next_pm = self.t_record.iloc[i_pm, -1] + self.th_time
                            if t_next_pm > self.tp:
                                self.tp_imported += 1
                                self.t_fault_list[i_pm] = self.life2[self.y2]
                                self.y2 += 1
                                self.t_record.iloc[i_pm, -1] = 0

                        else:
                            t_next_pm = self.t_record.iloc[i_pm, -1] + th_end - self.t_now_list[-1]
                            if t_next_pm > self.tp:
                                self.tp_homemade += 1
                                self.press_pm_in_low11()
                        # 判定是否会发生故障
                        t_record_temp = self.t_record.iloc[:, -1]
                        index_tf = [i for i in range(self.operation_number ) if
                                    self.t_fault_list[i] - t_record_temp[i] <= 0]
                        if len(index_tf) > 0:
                            self.tf_low_homemade += len(index_tf)
                            self.t_record.iloc[0, -1] = 0
                            self.t_fault_list[0] = self.life1[self.y1]
                            self.y1 += 1

                    self.press_low_to_high()
                    index_using = [0, 1]
                    dtf_high = [self.t_fault_list[i] - self.t_record.iloc[i, -1] for i in index_using]
                    dtf_high_min = min(dtf_high)
                    self.t_next_fault = self.t_now_list[-1] + dtf_high_min
                    while self.t_next_fault < self.th_end:
                        self.press_fault_in_high()

                    self.press_high_to_low1()
                    self.press_pm_small()

            self.tf_high_homemade_list.append(self.tf_high_homemade / self.cycle_times)
            self.tf_high_imported_list.append(self.tf_high_imported / self.cycle_times)
            self.tf_low__homemade_list.append(self.tf_low_homemade / self.cycle_times)
            self.tp_homemade_list.append(self.tp_homemade / self.cycle_times)
            self.tp_imported_list.append(self.tp_imported / self.cycle_times)

        cost_list = []
        for i in range(len(self.tp_list)):
            tf_high_homemade = self.tf_high_homemade_list[i] * self.cycle_times
            tf_high_imported = self.tf_high_imported_list[i] * self.cycle_times
            tf_low_homemade = self.tf_low__homemade_list[i] * self.cycle_times
            tp_homemade = self.tp_homemade_list[i] * self.cycle_times
            tp_imported = self.tp_imported_list[i] * self.cycle_times

            e1 = self.press_cost_high_fault(tf_high_homemade, tf_high_imported)
            e = e1 + self.press_cost_low_fault(tf_low_homemade) + self.press_cost_low_pm(tp_homemade, tp_imported)
            cost_list.append(e / self.cycle_times)

        self.verticalLayout_11.removeWidget(self.plt11)
        pg.setConfigOption("background", "w")
        self.plt11 = pg.PlotWidget()
        self.plt11.addLegend(size=(150, 80))
        self.plt11.showGrid(x=True, y=True, alpha=0.5)
        color = ["y", "b", "g"]
        label_list = ["方案11", "方案16", "方案17"]
        # for i in range(len(cost_list)):
        self.plt11.plot(x=self.tp_list, y=cost_list, pen=color[0], name=label_list[0])
        self.verticalLayout_11.addWidget(self.plt11)
        self.lineEdit_9.setText("方案11")
        self.lineEdit_10.setText("6700小时")

    def pump_strategy(self):
        self.verticalLayout_13.removeWidget(self.plt13)
        pg.setConfigOption("background", "w")
        self.plt13 = pg.PlotWidget()
        self.plt13.addLegend(size=(150, 80))
        self.plt13.showGrid(x=True, y=True, alpha=0.5)
        color = ["y", "b", "g"]
        label_list = ["方案11", "方案16", "方案17"]
        # for i in range(len(cost_list)):
        self.plt13.plot(x=np.random.rand(10), y=np.random.rand(10), pen=color[0], name=label_list[0])
        self.verticalLayout_13.addWidget(self.plt13)

    def press_strategy(self):
        self.verticalLayout_14.removeWidget(self.plt14)
        pg.setConfigOption("background", "w")
        self.plt14 = pg.PlotWidget()
        self.plt14.addLegend(size=(150, 80))
        self.plt14.showGrid(x=True, y=True, alpha=0.5)
        color = ["y", "b", "g"]
        label_list = ["方案11", "方案16", "方案17"]
        # for i in range(len(cost_list)):
        self.plt14.plot(x=np.random.rand(10), y=np.random.rand(10), pen=color[0], name=label_list[0])
        self.verticalLayout_14.addWidget(self.plt14)

    def pump_data(self):
        lmda1 = 1/self.doubleSpinBox.value()
        lmda2 = 1/self.doubleSpinBox_2.value()
        number_pump1 = self.spinBox.value()
        number_pump2 = self.spinBox_2.value()
        self.number_high = number_pump2 + number_pump1
        self.number_low = number_pump1
        self.number_minus = self.number_high - self.number_low

        self.pump_life2 = range(self.number_low,self.number_high)
        self.rate_self_maintenance = self.doubleSpinBox_4.value()
        self.rate_pm_small = self.doubleSpinBox_6.value()
        self.stop_day_self = self.doubleSpinBox_3.value()
        self.stop_day_out = self.doubleSpinBox_5.value()
        self.cost_pm_small = self.doubleSpinBox_11.value()
        self.cost_pm_large = self.doubleSpinBox_12.value()
        self.cost_fault = self.doubleSpinBox_15.value()
        self.cost_day = self.doubleSpinBox_14.value()

        self.life1 = []
        self.life2 = []
        self.rand1 = np.random.rand(300000)
        k_wbl = 5
        for i in range(len(self.rand1)):
            self.life1.append(sampling(i, self.rand1, lmda1, k_wbl))
            self.life2.append(sampling(i, self.rand1, lmda2, k_wbl))
        self.year = 25
        initial_time = 5880
        self.th_time = 2880
        th_start_list = [initial_time + i * 8760 for i in range(self.year)]
        th_end_list = [initial_time + self.th_time + i * 8760 for i in range(self.year)]
        self.t_table = pd.DataFrame()
        self.t_table["th_start"] = th_start_list
        self.t_table["th_end"] = th_end_list
        tp_list = np.linspace(6000, 14000, 11)

        self.tp_list = tp_list
        self.cost_list = []
        self.t_low_overlap_list = []
        self.tf_low_list = []
        self.tf_high_list = []
        self.tp_count_list = []
        self.Avai_low = []
        self.Avai_high = []
        self.cycle_times = 1

    def press_data(self):
        lmda1 = 1 / self.doubleSpinBox_7.value()
        lmda2 = 1 / self.doubleSpinBox_10.value()
        self.number_homemade = 1
        self.number_imported = 2
        self.operation_number = self.number_homemade + self.number_imported
        self.pump2_index = [1, 2]
        self.stop_day = self.doubleSpinBox_8.value()
        self.cost_pm_homemade = self.doubleSpinBox_16.value()
        self.cost_pm_imported = self.doubleSpinBox_13.value()
        self.cost_day = self.doubleSpinBox_9.value()
        self.homemade_time = self.stop_day*24
        self.cost_fault_homemade = self.doubleSpinBox_19.value()
        self.cost_fault_imported = self.doubleSpinBox_18.value()

        self.life1 = []
        self.life2 = []
        self.rand1 = np.random.rand(300000)
        k_wbl = 5
        for i in range(len(self.rand1)):
            self.life1.append(sampling(i, self.rand1, lmda1, k_wbl))
            self.life2.append(sampling(i, self.rand1, lmda2, k_wbl))

        year = 25
        initial_time = 5880
        self.th_time = 2880
        th_start_list = [initial_time + i * 8760 for i in range(year)]
        th_end_list = [initial_time + self.th_time + i * 8760 for i in range(year)]
        self.t_table = pd.DataFrame()
        self.t_table["th_start"] = th_start_list
        self.t_table["th_end"] = th_end_list
        tp_list = np.linspace(6000, 14000, 161)
        self.tp_list = tp_list
        self.tf_low__homemade_list = []
        self.tp_homemade_list = []
        self.tp_imported_list = []
        self.tf_high_homemade_list = []
        self.tf_high_imported_list = []
        self.cycle_times = 1


    def pump_fault_in_low(self):
        index_spare = [i for i in range(self.number_high) if self.s_record.iloc[i, -1] != 1]
        t_fault_spare = [self.t_fault_list[i] for i in index_spare]
        index_using = [i for i in range(self.number_high) if self.s_record.iloc[i,-1] ==1]
        t_fault_temp = self.t_fault_list.copy()
        for i in index_spare:
            t_fault_temp[i] = 1000000
        dtf = [t_fault_temp[i] - self.t_record.iloc[i,-1] for i in range(self.number_high)]
        dtf_min = min(dtf)
        index_fault = dtf.index(min(dtf))
        t_now_fault = self.t_now_list[-1] + dtf_min
        self.t_now_list.append(t_now_fault)
        if index_fault in self.pump_life2:
            self.t_fault_list[index_fault] = self.life2[self.y2]
            self.y2 += 1
        else:
            self.t_fault_list[index_fault] = self.life1[self.y1]
            self.y1 += 1

        if self.low_change == 1:
            index_change = index_spare[t_fault_spare.index(min(t_fault_spare))]
        else:
            index_change = index_spare[t_fault_spare.index(max(t_fault_spare))]
        s_record_now = [self.s_record.iloc[i,-1] for i in range(self.number_high)]
        s_record_now[index_fault] = 2
        s_record_now[index_change] = 1
        self.s_record[self.x+1] = s_record_now
        t_record_now = [self.t_record.iloc[i,-1] for i in range(self.number_high)]
        for i in index_using:
            t_record_now[i] += dtf_min
        t_record_now[index_fault] = 0
        self.t_record[self.x+ 1] = t_record_now
        self.x += 1

    def pump_pm_in_low(self):
        index_spare = [i for i in range(self.number_high) if self.s_record.iloc[i,-1] != 1]
        t_operation_spare = [self.t_record.iloc[i,-1] for i in index_spare]
        index_using = [i for i in range(self.number_high) if self.s_record.iloc[i,-1] ==1]
        dtp = [self.tp - self.t_record.iloc[i,-1] for i in range(self.number_high)]
        for i in index_spare:
            dtp[i] = 1000000
        dtp_min =min(dtp)
        index_min_pm = dtp.index(min(dtp))
        t_next_pm = self.t_now_list[-1] + dtp_min
        self.t_now_list.append(t_next_pm)
        if index_min_pm in self.pump_life2:
            self.t_fault_list[index_min_pm] = self.life2[self.y2]
            self.y2 += 1
        else:
            self.t_fault_list[index_min_pm] = self.life1[self.y1]
            self.y1 += 1
        if self.low_change == 1:
            index_change = index_spare[t_operation_spare.index(min(t_operation_spare))]
        else:
            index_change = index_spare[t_operation_spare.index(max(t_operation_spare))]
        s_record_now = [self.s_record.iloc[i,-1] for i in range(self.number_high)]
        s_record_now[index_min_pm] = 3
        s_record_now[index_change] = 1
        self.s_record[self.x + 1] = s_record_now
        t_record_now = [self.t_record.iloc[i,-1] for i in range(self.number_high)]
        for i in index_using:
            t_record_now[i] += dtp_min
        t_record_now[index_min_pm] = 0
        self.t_record[self.x+1] = t_record_now
        self.x+=1

    def pump_low_to_high(self):
        dt = self.th_start - self.t_now_list[-1]
        self.t_now_list.append(self.th_start)
        index_spare = [i for i in range(self.number_high) if self.s_record.iloc[i,-1] != 1]
        t_record_now = [self.t_record.iloc[i,-1] + dt for i in range(self.number_high)]
        for i in index_spare:
            t_record_now[i] = self.t_record.iloc[i,-1]
        self.s_record[self.x+1] = [1 for i in range(self.number_high)]
        self.t_record[self.x+1] = t_record_now
        self.x += 1

    def pump_fault_in_high(self):
        dtf = [self.t_fault_list[i] - self.t_record.iloc[i,-1] for i in range(self.number_high)]
        dtf_min = min(dtf)
        index_fault = dtf.index(min(dtf))
        t_now_fault = self.t_now_list[-1] + dtf_min
        self.t_now_list.append(t_now_fault)
        if index_fault in self.pump_life2:
            self.t_fault_list[index_fault] = self.life2[self.y2]
            self.y2+= 1
        else:
            self.t_fault_list[index_fault] = self.life1[self.y1]
            self.y1+=1
        s_record_now = [i for i in range(self.number_high)]
        s_record_now[index_fault] = 2
        self.s_record[self.x+1] = s_record_now
        t_record_now = [self.t_record.iloc[i,-1] + dtf_min for i in range(self.number_high)]
        t_record_now[index_fault] = 0
        self.t_record[self.x+1] = t_record_now
        self.x += 1
        dtf = [self.t_fault_list[i] - self.t_record.iloc[i,-1] for i in range(self.number_high)]
        dtf_min = min(dtf)
        self.t_next_fault = self.t_now_list[-1] + dtf_min

    def pump_high_to_low(self):
        t_now = self.t_now_list[-1]
        self.t_now_list.append(self.th_end)
        dt = self.th_end - t_now
        t_record_now = [self.t_record.iloc[i,-1] + dt for i in range(self.number_high)]
        t_record_temp = t_record_now.copy()
        index_stop = []
        if self.high_to_low_number  == 1: #停掉运行时间长的设备
            for i in range(self.number_minus):
                index1 = t_record_temp.index(max(t_record_temp))
                index_stop.append(index1)
                t_record_temp[index1] = -1
        elif self.high_to_low_number == 2: #停掉运行时间短的设备
            for i in range(self.number_minus):
                index1 = t_record_temp.index(min(t_record_temp))
                index_stop.append(index1)
                t_record_temp[index1] = -1
        elif self.high_to_low_number == 3: #停掉中间的设备
            index_start = self.number_low // 2
            for i in range(self.number_minus):
                index_stop.append(index_start+i)

        elif self.high_to_low_number == 4: #顺序停机
            index_start = (self.m%(self.number_high//self.number_minus))*self.number_minus
            for i in range(self.number_minus):
                index_stop.append(index_start+i)
        elif self.high_to_low_number == 5: #现有规则停机
            for i in range(self.number_minus):
                index1 = self.get_index(t_record_temp)
                t_record_temp[index1] = 1000000
                index_stop.append(index1)
        elif self.high_to_low_number == 6: #偏好停掉第二类的泵体
            for i in range(self.number_minus):
                index1 = self.pump_life2[i]
                index_stop.append(index1)
        else:
            for i in range(self.number_minus):
                index_stop.append(i)

        self.t_record[self.x+1] = t_record_now
        s_record_now = [1 for i in range(self.number_high)]
        for i in index_stop:
            s_record_now[i] = 3
        self.s_record[self.x+1] = s_record_now
        self.x+=1

    def pump_cost_low_pm(self,number):
        cost_rand = self.cost_rand[:number]
        large = sum(cost_rand>self.rate_pm_small)
        cost_sum = large*self.cost_pm_large
        cost_sum += (number-large)*self.cost_pm_small
        return cost_sum

    def pump_cost_high_fault(self,number):
        cost_rand = self.cost_rand[:number]
        large = sum(cost_rand>self.rate_self_maintenance)
        cost_sum = large*(self.cost_fault+self.cost_day*self.stop_day_out)
        cost_sum += (number-large)*(self.cost_fault+self.cost_day*self.stop_day_self)
        return cost_sum

    def pump_cost_low_fault(self,number):
        cost = number*self.cost_fault
        return cost

    def pump_cost_downtime(self,hours):
        day = hours/24
        cost = day*self.cost_day
        return cost

    def press_fault_in_low11(self):
        t_record_temp = self.t_record.iloc[:, -1].tolist()
        dt = self.t_fault_list[0] - self.t_record.iloc[0, -1]
        self.t_now_list.append(self.t_now_list[-1] + dt)
        t_record_temp[0] = 0
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = [2, 1, 2]
        self.x += 1

        self.t_now_list.append(self.t_now_list[-1] + 240)
        t_record_temp = self.t_record.iloc[:, -1] + 240
        t_record_temp[0] = 0
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = [1, 3, 3]
        self.x += 1
        self.t_fault_list[0] = self.life1[self.y1]
        self.y1 += 1

    def press_pm_in_low11(self):
        t_record_temp = self.t_record.iloc[:, -1].tolist()
        dt = self.tp - self.t_record.iloc[0, -1]
        self.t_now_list.append(self.t_now_list[-1] + dt)
        t_record_temp[0] = 0
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = [4, 1, 1]
        self.x += 1

        dt = 240
        self.t_now_list.append(self.t_now_list[-1] + dt)
        t_record_temp = self.t_record.iloc[:, -1] + dt
        t_record_temp[0] = 0
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = [1, 3, 3]
        self.x += 1
        self.t_fault_list[0] = self.life1[self.y1]
        self.y1 += 1

    def press_pm_small(self):
        self.t_now_list.append(self.t_next_fault)
        self.t_fault_list[0] += 300
        t_record_temp = self.t_record.iloc[:, -1].tolist()
        t_record_temp[0] = (t_record_temp[0] // 3000 + 1) * 3000
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = self.s_record[self.x]
        self.x += 1

    def press_low_to_high(self):
        dt = self.th_start - self.t_now_list[-1]
        self.t_now_list.append(self.th_start)
        t_record_temp = self.t_record.iloc[:, -1].tolist()
        t_record_temp[0] += dt
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = [1, 1, 3]
        self.x += 1

    def press_fault_in_high(self):
        index_using = [i for i in range(self.operation_number) if self.s_record.iloc[i, -1] == 1]
        dtf = [self.t_fault_list[i] - self.t_record.iloc[i, -1] for i in index_using]
        dtf_min = min(dtf)
        index_fault = index_using[dtf.index(min(dtf))]
        t_now_fault = self.t_now_list[-1] + dtf_min
        self.t_now_list.append(t_now_fault)
        s_record_temp = [1 for i in range(self.operation_number)]
        s_record_temp[index_fault] = 2
        self.s_record[self.x + 1] = s_record_temp
        t_record_temp = self.t_record.iloc[:, -1].tolist()
        for i in index_using:
            t_record_temp[i] += dtf_min
        t_record_temp[index_fault] = 0
        self.t_record[self.x + 1] = t_record_temp
        self.x += 1
        if index_fault in self.pump2_index:
            self.tf_high_imported += 1
            self.t_fault_list[index_fault] = self.life2[self.y2]
            self.y2 += 1
        else:
            self.tf_high_homemade += 1
            self.t_fault_list[index_fault] = self.life1[self.y1]
            self.y1 += 1

            self.t_now_list.append(self.t_now_list[-1] + 240)
            index_using = [i for i in range(self.operation_number) if self.s_record.iloc[i, -1] == 1]
            for i in index_using:
                t_record_temp[i] += 240
            self.t_record[self.x + 1] = t_record_temp
            self.s_record[self.x + 1] = [1, 1, 3]
            self.x += 1
        index_using = [i for i in range(self.operation_number) if self.s_record.iloc[i, -1] == 1]
        dtf = [self.t_fault_list[i] - self.t_record.iloc[i, -1] for i in index_using]
        self.t_next_fault = self.t_now_list[-1] + min(dtf)

    def press_high_to_low1(self):
        index_using = [i for i in range(self.operation_number) if self.s_record.iloc[i, -1] == 1]
        dt = self.th_end - self.t_now_list[-1]
        self.t_now_list.append(self.th_end)
        t_record_temp = self.t_record.iloc[:, -1].tolist()
        for i in index_using:
            t_record_temp[i] += dt
        self.t_record[self.x + 1] = t_record_temp
        self.s_record[self.x + 1] = [1, 3, 3]
        self.x += 1

    def press_cost_low_pm(self,home_number,imported_number):
        cost = self.cost_pm_homemade * home_number + self.cost_pm_imported * imported_number
        return cost

    def press_cost_high_fault(self,home_number, imported_number):
        cost = home_number * (
                    self.cost_pm_homemade + self.cost_day * self.stop_day) + imported_number * self.cost_pm_imported
        return cost

    def press_cost_low_fault(self,number):
        cost = self.cost_pm_homemade * number
        return cost

    def press_cost_downtime(self, hours):
        day = hours/24
        cost = day*self.cost_day
        return cost

    def pump_low_cb(self, cb):
        # 复选框内容添加
        if cb.isChecked():
            if cb.text()[0] not in self.pump_low_items:
                self.pump_low_items.append(cb.text()[0])

            shop_cart= ",".join(self.pump_low_items)
            self.lineEdit_2.setText(shop_cart)
        else:
            if cb.text()[0] in self.pump_low_items:
                self.pump_low_items.remove(cb.text()[0])

            shop_cart = ",".join(self.pump_low_items)
            self.lineEdit_2.setText(shop_cart)

    def pump_high_cb(self, cb):
        # 复选框内容添加
        if cb.isChecked():
            if cb.text()[0] not in self.pump_high_items:
                self.pump_high_items.append(cb.text()[0])
            shop_cart= ",".join(self.pump_high_items)
            self.lineEdit.setText(shop_cart)
        else:
            if cb.text()[0] in self.pump_high_items:
                self.pump_high_items.remove(cb.text()[0])

            shop_cart = ",".join(self.pump_high_items)
            self.lineEdit.setText(shop_cart)

    def press_low_cb(self, cb):
        # 复选框内容添加
        if cb.isChecked():
            if cb.text()[0] not in self.press_low_items:
                self.press_low_items.append(cb.text()[0])
            shop_cart= ",".join(self.press_low_items)
            self.lineEdit_4.setText(shop_cart)
        else:
            if cb.text()[0] in self.press_low_items:
                self.press_low_items.remove(cb.text()[0])
            shop_cart = ",".join(self.press_low_items)
            self.lineEdit_4.setText(shop_cart)
    def press_high_cb(self, cb):
        # 复选框内容添加
        if cb.isChecked():
            if cb.text()[0] not in self.press_high_items:
                self.press_high_items.append(cb.text()[0])

            shop_cart= ",".join(self.press_high_items)
            self.lineEdit_3.setText(shop_cart)
        else:
            if cb.text()[0] in self.press_high_items:
                self.press_high_items.remove(cb.text()[0])

            shop_cart = ",".join(self.press_high_items)
            self.lineEdit_3.setText(shop_cart)

if __name__ == '__main__':
    cgitb.enable(format="text")
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
