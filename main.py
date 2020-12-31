import sys
import pandas as pd
import numpy as np
import math
import pyqtgraph as pg
import cgitb
from PyQt5.Qt import QThread
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow
import LNG_demo_1229



def sampling(Rf, lmda, k_wbl):
    """
    蒙特卡洛逆变换抽样原理，类似于一个反函数求解
    Params：
        Rf:  01之间的随机数
        lmda: 尺度参数
        k_wbl:  形状参数
    return:
        tf 满足威布尔分布的随机数
    """
    tf = (-math.log(1 - Rf)) ** (1 / k_wbl) / lmda
    return tf


def mean_life(t):
    """
    给定设计寿命，求解平均寿命的函数，假定是平均可用度为99%时对应的运行时间为设计寿命
    Params：
        t 设计寿命
    Return：
        T 平均寿命
    """
    T = t / 0.04 ** (1 / 3)
    return T


def cost_pm(idx_pm, tmp_count):
    """
    一次PM的费用是固定的
    params：
        idx_pm：预防维修的设备的编号
        tmp_count:用以记录常数的字典
    return：
        不同设备一次PM对应的费用
    """
    return tmp_count[idx_pm]['c_pm']


def cost_cm(idx_fault, tmp_count):
    """
    一次CM的费用是固定的
    params：
        idx_fault：预防维修的设备的编号
        tmp_count:用以记录常数的字典
    return：
        不同设备一次CM对应的费用
    """
    return tmp_count[idx_fault]['c_cm']


def cost_stop(idx, tmp_count):
    """
    一次产能损失的费用是固定的
    params：
        idx：预防维修的设备的编号
        tmp_count:用以记录常数的字典
    return：
        不同设备一次产能损失对应的费用
    """
    return tmp_count[idx]['cs']


def pump_thread(Qthread):
    


class demo(QMainWindow, LNG_demo_1229.Ui_mainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        LNG_demo_1229.Ui_mainWindow.__init__(self)
        self.setupUi(self)
        self.interface_plus()
        self.pump_params()
        self.pushButton_1.clicked.connect(self.pump_drawing)
        self.pushButton_2.clicked.connect(self.press_drawing)


    def pump_drawing(self):
        """
        首先对图像进行修改，然后把计算结果绘制成图像，放在绘图区上
        :return:
        """
        self.verticalLayout.removeWidget(self.plt)
        pg.setConfigOption("background", "w")
        self.plt = pg.PlotWidget()
        pltItem = self.plt.getPlotItem()
        left_axis = pltItem.getAxis("left")
        left_axis.enableAutoSIPrefix(False)
        font = QFont()
        font.setPixelSize(16)
        left_axis.tickFont = font
        bottom_axis = pltItem.getAxis("bottom")
        bottom_axis.tickFont = font
        labelStyle = {'color': '#000', 'font-size': '16pt'}
        left_axis.setLabel('维修总花费', units='万元', **labelStyle)
        bottom_axis.setLabel('维修周期', units='天', **labelStyle)


        self.plt.addLegend(size=(150, 50))
        self.plt.showGrid(x=True, y=True, alpha=0.5)
        self.pump_count()
        x_num = [int(x) for x in self.all_record[0].index]
        y_num = 0
        for i in range(len(self.all_record)):
            y_num += self.all_record[i]['sum_cost'] / self.times

        self.plt.plot(x=x_num, y=y_num, name='高压泵预防维修成本曲线', pen='r')

        min_fare = min(y_num)
        y_num = y_num.tolist()
        min_d = x_num[y_num.index(min_fare)]
        self.lineEdit_1.setText(str(min_d)+'天')
        self.lineEdit_2.setText(str(min_fare)+'万元')
        self.verticalLayout.addWidget(self.plt)

    def pump_params(self):
        """
        读取所有的高压泵的参数
        :return:
        """
        life1_value = self.doubleSpinBox_1.value()/24
        life2_value = self.doubleSpinBox_2.value()/24
        mean_home = mean_life(life1_value)
        mean_import = mean_life(life2_value)  # 进口高压泵的设计寿命时10000h，也就是417天
        np.random.seed(1)
        rand_list = np.random.rand(1000)  # 随机数
        k_wbl = 3
        life1 = [sampling(i, 1 / mean_home, k_wbl) for i in rand_list]
        life2 = [sampling(i, 1 / mean_import, k_wbl) for i in rand_list]

        # 如果要做扩展，就需要对每个参数单独输入
        self.num_home = self.spinBox_1.value()
        self.num_import = self.spinBox_2.value()
        self.num_high = self.num_home + self.num_import
        self.num_low = self.spinBox_3.value()
        dic_import = {'c_pm': self.doubleSpinBox_3.value(),  # 国产pump一次pm的固定费用
                      'c_cm': self.doubleSpinBox_4.value(),  # cm的费用
                      'cs': self.doubleSpinBox_5.value(),  # 单次产能损失
                      'mean_life': mean_import,
                      'life': life2}
        dic_home = {'c_pm': self.doubleSpinBox_6.value(), #进口pump一次pm的固定费用
                    'c_cm': self.doubleSpinBox_7.value(),
                    'cs': self.doubleSpinBox_8.value(),
                    'mean_life': mean_home,
                    'life': life1}
        self.high_change = 1
        # 高负荷期的长度也是可以调节
        d_year = 360
        self.th_time = 30 * self.spinBox_4.value() #可以改变

        pump_tmp = {}
        for idx in range(self.num_home):
            pump_tmp[idx] = dic_home.copy()
            pump_tmp[idx]['life'] = life2.copy()[:-(idx + 1) * 10]

        for idx in range(self.num_home, self.num_home + self.num_import):
            pump_tmp[idx] = dic_import.copy()
            pump_tmp[idx]['life'] = life1.copy()[:-(idx + 1) * 10]
        pump_tmp['h_time'] = self.th_time
        self.pump_tmp = pump_tmp

        initial_time = d_year - self.th_time
        year = 25
        th_start_list = [initial_time + i * d_year for i in range(year)]
        th_end_list = [(i + 1) * d_year for i in range(year)]
        t_table = pd.DataFrame()
        t_table['th_start'] = th_start_list
        t_table['th_end'] = th_end_list
        self.t_table = t_table

        cols = ['n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'sum_cost']
        self.times = int(self.spinBox_9.value())
        low = int(self.spinBox_10.value())
        high = int(self.spinBox_11.value()) + 1
        bin = int(self.spinBox_12.value())
        range1 = range(low, high, bin)
        self.range1 = range1
        idxs = [str(tp) for tp in range1]
        df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
        all_record = {}
        for idx in range(self.num_home + self.num_import):
            all_record[idx] = df.copy()

        self.all_record = all_record
        self.col_list = ['t1', 't2', 'state', 'life', 'n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'now',
                         'sum_cost']

    def pump_init_proc(self, dic, tp):
        """
        高压泵低负荷期下一次维护的操作类型与实践间隔
        params:
            dic:记录所有的运行信息
            tp： 维修周期
        return
            flag， dt
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_p)
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        if dt < dtf:
            flag = 2
        else:
            flag = 1
            dt = dtf
        return flag, dt

    def pump_init_high(self, dic, tmp_count):
        """
        高负荷期内判定下一次操作的时间
        params:
            dic:记录所有的运行信息
            tp： 维修周期
        return
            dt 下一次故障间隔
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_f)
        return dt

    def pump_low_fault(self, dic, tmp_count):
        """
            低负荷期发生故障，拟考虑运行时间较短的备件来接替运行
            params:
                dic:记录信息的字典,改变
                tmp_count:设备固有数据，不会变
            return:
                dic
            """

        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        # 不管怎么样，以下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了,df自动更新的
            # 只考虑fault只有一台机器损坏
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'state'] = 3
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_cm']

        idx_spare = [i for i in range(len(dic) - 1) if i not in idx_using]
        idx_u = self.pump_low_change(dic, idx_spare)  # 就考虑运行时间短的设备来接替工作
        dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        return dic

    def pump_low_change(self, dic, idx_spare):
        """
            找到运行时间最低的备件来接替运行
            params：
                dic:运行记录的字典
                idx_spare:备件的编号
            return：
                idx
            """
        oper_time = [dic[i].loc[dic[i].index[-1], 't1'] for i in idx_spare]
        idx = idx_spare[oper_time.index(min(oper_time))]
        return idx

    def pump_low_pm(self, dic, tp, tmp_count):
        """
        低负荷期发生预防性维修的函数，拟采用运行时间较少的备件来接替运行
        同时考虑多台设备一起维修的情况
        params:
            dic:记录信息的字典
            tp:维修周期
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtp = min(t_p)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
        idx_pm = []
        for idx_ in range(len(idx_using)):
            if t_p[idx_] == dtp:
                idx_pm.append(idx_using[idx_])
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_ in idx_pm:
            x = dic[idx_].index[-1]  # 这个时候x已经自加了
            dic[idx_].loc[x, 't1'] = 0
            dic[idx_].loc[x, 'state'] = 3
            dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
            dic[idx_].loc[x - 1, 'n_pm'] = 1
            dic[idx_].loc[x - 1, 'c_pm'] = cost_pm(idx_, tmp_count)
            dic[idx_].loc[x - 1, 'sum_cost'] += dic[idx_].loc[x - 1, 'c_pm']

            # 这个得分情况讨论吧，如果同时维修数大于2，则直接原地维修
        # 如果小于2，则可以考虑接替
        idx_spare = [i for i in range(len(dic) - 1) if i not in idx_using]
        n_pm = len(idx_pm)
        if n_pm > 2:
            for idx_u in idx_pm:
                dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        elif n_pm == 2:
            idx_using = [i for i in range(len(dic) - 1) if i not in idx_pm]
            for idx_u in idx_using:
                dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        else:
            idx_u = self.pump_low_change(dic, idx_spare)
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        return dic

    def pump_l2h(self, dic, tp, tmp_count, t_next_start):
        """
        设备从低负荷期到高负荷期的转换过程，所有设备都处于工作状态
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
            t_next_start: 高负荷期开始的时间
            tp:维修周期
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_start - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_start)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1  # 先进入停机状态，然后再确定哪些机器需要继续运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        idx_using = range(len(dic) - 1)
        # 判定是否会在高负荷期内达到预防周期，是则提前进行预防维修，否则不用操作
        for idx_u in idx_using:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
            if dic[idx_u].loc[dic[idx_u].index[-1], 't1'] + tmp_count['h_time'] >= tp:
                x = dic[idx_u].index[-1]
                dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1']
                dic[idx_u].loc[x + 1, 't1'] = 0  # 因为进行了pm，所以下一阶段的t1一定是0
                dic[idx_u].loc[x + 1, 'life'] = tmp_count[idx_u]['life'].pop()
                dic[idx_u].loc[x + 1, 'state'] = 1
                dic[idx_u].loc[x, 'n_pm'] = 1
                dic[idx_u].loc[x, 'n_cm'] = 0
                dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
                dic[idx_u].loc[x, 'c_cm'] = 0
                dic[idx_u].loc[x, 'c_stop'] = 0
                dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
                dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_pm']
        return dic

    def pump_high(self, dic, tmp_count):
        """
        设备在高负荷期运行的模式，设备发生故障，产生产能损失，修复完成后，立即投入使用
        params：
            dic:记录信息的字典
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        # 不管怎么样，一下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1  # 先进入停机状态，然后再令0号机开始运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'c_stop'] = cost_stop(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_stop'] + dic[idx_f].loc[x - 1, 'c_cm']
        return dic

    def pump_high_change2(self, dic, num):
        """
        策略2：停掉运行时间最长的两台高压泵，对比组
        params:
            dic:记录设备运行信息的字典，可变
            num：需要停掉的设备数目
        return:
            idx：num个需要停机的设备的编号
        """
        life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic) - 1)]
        life_list = np.array(life_list)  # 必须是数组形式，才能使用负号输出
        sorted_list = np.argsort(-life_list)
        return sorted_list[:num]

    def pump_high_change1(self, dic, num):
        """
        策略1：停掉距离最近的num台泵,正常组
        params:
            dic:记录设备运行信息的字典，可变
            num：需要停掉的设备数目
        return:
            idx：num个需要停机的设备的编号
        """
        life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic) - 1)]
        sorted_nums = sorted(enumerate(life_list), key=lambda x: x[1])
        df = pd.DataFrame(sorted_nums, columns=['idx', 'num'])
        df['diff'] = df['num'].diff()  # 累减函数，从小往上1个步长
        df['diff'] = df['diff'].fillna(1000000)  # 这就保证了运行时间最小的设备一定不会停机
        df = df.sort_values(by="diff", axis=0, ascending=True)  # axis=0就是表示的一列
        range1 = df['idx'][:num]
        return range1

    def pump_h2l(self, dic, tmp_count, t_next_end, high_change, num):
        """
        设备从高负荷期向低负荷期切换的策略，这个策略比较多
        params
            dic:记录设备运行信息
            tmp_count:设备不变的信息
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_end - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_end)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        if high_change == 1:
            # 正常组，LNG现有策略
            idx_list = self.pump_high_change1(dic, num)
        else:
            # 对比组，停掉运行时间最长的两台设备
            idx_list = self.pump_high_change2(dic, num)

        for idx in idx_list:
            dic[idx].loc[dic[idx].index[-1], 'state'] = 3
        return dic

    def pump_count(self):
        """
        高压泵主题计算过程，包括读入数据，得到最后的计算结果
        :return:
        """
        self.pump_params()
        for tp in tqdm(self.range1):
            for _ in range(self.times):
                df_ = pd.DataFrame([[0] * len(self.col_list)], columns=self.col_list)
                dic = {'t_now_list': [0]}
                for idx_machine in range(self.num_home + self.num_import):
                    # 建立所有设备的运行记录时间
                    dic[idx_machine] = df_.copy()
                    dic[idx_machine].loc[dic[idx_machine].index[-1], 'life'] = self.pump_tmp[idx_machine]['life'].pop()
                for idx in range(self.num_low):
                    dic[idx]['state'] = 1

                for idx_t in range(len(self.t_table)):
                    t_next_start = self.t_table.iloc[idx_t, 0]
                    flag, dt = self.pump_init_proc(dic, tp)
                    while dic['t_now_list'][-1] + dt < t_next_start:
                        if flag == 1:
                            dic = self.pump_low_fault(dic, self.pump_tmp)
                        else:
                            dic = self.pump_low_pm(dic, tp, self.pump_tmp)
                        flag, dt = self.pump_init_proc(dic, tp)

                    dic = self.pump_l2h(dic, tp, self.pump_tmp, t_next_start)
                    t_next_end = self.t_table.iloc[idx_t, 1]
                    dt = self.pump_init_high(dic, self.pump_tmp)  # 只需要考虑哪个会先到平均寿命节点
                    while dic['t_now_list'][-1] + dt < t_next_end:
                        dic = self.pump_high(dic, self.pump_tmp)
                        dt = self.pump_init_high(dic, self.pump_tmp)
                    num = 2
                    dic = self.pump_h2l(dic, self.pump_tmp, t_next_end, self.high_change, num)

                for idx_machine in range(len(dic) - 1):
                    self.all_record[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
                    self.all_record[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['c_cm'])
                    self.all_record[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
                    self.all_record[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])

        for idx_machine in range(len(dic) - 1):
            self.all_record[idx_machine]['n_stop'] = self.all_record[idx_machine]['c_stop'] / \
                                                     self.pump_tmp[idx_machine]['cs']

    def interface_plus(self):
        """
        用绘图先占据龚白的部分，使图像不至于突兀
        :return:
        """
        pg.setConfigOption("background", "w")
        self.plt = pg.PlotWidget()
        self.plt.addLegend(size=(150, 50))
        self.plt.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout.addWidget(self.plt)

        self.plt2 = pg.PlotWidget()
        self.plt2.addLegend(size=(150, 50))
        self.plt2.showGrid(x=True, y=True, alpha=0.5)
        self.verticalLayout_2.addWidget(self.plt2)



    def press_drawing(self):
        """
        首先对图像进行修改，然后把计算结果绘制成图像，放在绘图区上
        :return:
        """
        self.verticalLayout_2.removeWidget(self.plt2)
        pg.setConfigOption("background", "w")
        self.plt2 = pg.PlotWidget()
        pltItem = self.plt2.getPlotItem()
        left_axis = pltItem.getAxis("left")
        left_axis.enableAutoSIPrefix(False)
        font = QFont()
        font.setPixelSize(16)
        left_axis.tickFont = font
        bottom_axis = pltItem.getAxis("bottom")
        bottom_axis.tickFont = font
        labelStyle = {'color': '#000', 'font-size': '16pt'}
        left_axis.setLabel('维修总花费', units='万元', **labelStyle)
        bottom_axis.setLabel('维修周期', units='天', **labelStyle)

        self.plt2.addLegend(size=(150, 50))
        self.plt2.showGrid(x=True, y=True, alpha=0.5)
        self.press_count()
        x_num = [int(x) for x in self.all_record_press[0].index]
        y_num = 0
        for i in range(len(self.all_record_press)):
            y_num += self.all_record_press[i]['sum_cost'] / self.times

        self.plt2.plot(x=x_num, y=y_num, name='压缩机预防维修成本曲线', pen='r')

        min_fare = min(y_num)
        y_num = y_num.tolist()
        min_d = x_num[y_num.index(min_fare)]
        self.lineEdit_3.setText(str(min_d) + '天')
        self.lineEdit_4.setText(str(min_fare) + '万元')
        self.verticalLayout_2.addWidget(self.plt2)


    def press_params(self):
        """
        读取所有的压缩机的参数
        """
        life1_value = self.doubleSpinBox_15.value() / 24
        life2_value = self.doubleSpinBox_16.value() / 24
        mean_home = mean_life(life1_value)
        mean_import = mean_life(life2_value)  # 进口高压泵的设计寿命时10000h，也就是417天
        np.random.seed(1)
        rand_list = np.random.rand(1000)  # 随机数
        k_wbl = 3
        life1 = [sampling(i, 1 / mean_home, k_wbl) for i in rand_list]
        life2 = [sampling(i, 1 / mean_import, k_wbl) for i in rand_list]

        # 如果要做扩展，就需要对每个参数单独输入
        self.num_home_press = self.spinBox_7.value()
        self.num_import_press = self.spinBox_8.value()
        self.num_high_press = self.num_home_press + self.spinBox_17.value()
        self.num_low_press = self.spinBox_5.value()
        dic_import = {'c_pm': self.doubleSpinBox_9.value(),  # 国产press一次pm的固定费用
                      'c_cm': self.doubleSpinBox_10.value(),  # cm的费用
                      'cs': self.doubleSpinBox_11.value(),  # 单次产能损失
                      'mean_life': mean_import,
                      'life': life2}
        dic_home = {'c_pm': self.doubleSpinBox_12.value(),  # 进口press一次pm的固定费用
                    'c_cm': self.doubleSpinBox_13.value(),
                    'cs': self.doubleSpinBox_14.value(),
                    'mean_life': mean_home,
                    'life': life1}

        # 高负荷期的长度也是可以调节
        d_year = 360
        self.th_time_press = 30 * self.spinBox_6.value()  # 可以改变

        press_tmp = {} # 记录不变数据的字典
        for idx in range(self.num_home_press):
            press_tmp[idx] = dic_home.copy()
            press_tmp[idx]['life'] = life1.copy()[:-(idx + 1) * 10]

        for idx in range(self.num_home_press, self.num_home_press + self.num_import_press):
            press_tmp[idx] = dic_import.copy()
            press_tmp[idx]['life'] = life2.copy()[:-(idx + 1) * 10]
        press_tmp['h_time'] = self.th_time_press
        self.press_tmp = press_tmp

        initial_time = d_year - self.th_time_press
        year = 25
        th_start_list = [initial_time + i * d_year for i in range(year)]
        th_end_list = [(i + 1) * d_year for i in range(year)]
        t_table = pd.DataFrame()
        t_table['th_start'] = th_start_list
        t_table['th_end'] = th_end_list
        self.t_table_press = t_table

        cols = ['n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'sum_cost']
        self.times = int(self.spinBox_13.value())
        low = int(self.spinBox_14.value())
        high = int(self.spinBox_15.value()) + 1
        bin = int(self.spinBox_16.value())
        range2 = range(low, high, bin)
        self.range2 = range2
        idxs = [str(tp) for tp in range2]
        df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
        all_record = {}
        for idx in range(self.num_home_press + self.num_import_press):
            all_record[idx] = df.copy()

        self.all_record_press = all_record
        self.col_list = ['t1', 't2', 'state', 'life', 'n_pm', 'n_cm', 'n_stop', 'c_pm', 'c_cm', 'c_stop', 'now',
                         'sum_cost']

    def press_init_proc(self, dic, tp):
        """
        低负荷期判断下一次维护是什么时候
        params:
            dic:记录所有的运行信息
            tp： 维修周期
        return
            flag， dt
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_p)
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        if dt < dtf:
            flag = 2
        else:
            flag = 1
            dt = dtf
        return flag, dt

    def press_init_high(self, dic, tp):
        """
        高负荷期内发生故障的维护程序
        params：
            dic:记录所有的运行信息
            tp： 维修周期
        return
            dt 下一次故障间隔
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dt = min(t_f)
        return dt

    def press_low_fault(self, dic, tmp_count):
        """
        低负荷期发生故障，拟考虑运行时间较短的备件来接替运行
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
        return:
            dic
        """

        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        # 不管怎么样，以下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了,df自动更新的
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_cm']
        return dic

    def press_low_pm(self, dic, tp, tmp_count):
        """
        低负荷期发生预防性维修的函数，拟采用运行时间较少的备件来接替运行
        同时考虑多台设备一起维修的情况
        params:
            dic:记录信息的字典
            tp:维修周期
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtp = min(t_p)
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
        idx_pm = []
        for idx_ in range(len(idx_using)):
            if t_p[idx_] == dtp:
                idx_pm.append(idx_using[idx_])
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        for idx_ in idx_pm:
            x = dic[idx_].index[-1]  # 这个时候x已经自加了
            dic[idx_].loc[x, 't1'] = 0
            dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
            dic[idx_].loc[x - 1, 'n_pm'] = 1
            dic[idx_].loc[x - 1, 'c_pm'] = cost_pm(idx_, tmp_count)
            dic[idx_].loc[x - 1, 'sum_cost'] = dic[idx_].loc[x - 1, 'c_pm']
        return dic

    def press_l2h(self, dic, tp, tmp_count, t_next_start):
        """
        设备从低负荷期到高负荷期的转换过程，所有设备都处于工作状态
        params:
            dic:记录信息的字典,改变
            tmp_count:设备固有数据，不会变
            t_next_start: 高负荷期开始的时间
            tp:维修周期
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_start - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_start)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 3  # 先进入停机状态，然后再确定哪些机器需要继续运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0
        idx_using = [idx_home for idx_home in range(self.num_home_press)] #这里需要重新写,有可能国产是几台压缩机
        #对进口压缩机的运行时间进行排序，取最小的 x 台
        range_tmp = range(self.num_home_press, self.num_home_press+self.num_import_press)
        idx_list = list(range_tmp)
        using_time = [dic[i].loc[dic[i].index[-1], 't1'] for i in range_tmp]
        using_time = list(enumerate(using_time))
        sorted_time = sorted(using_time, key=lambda x: x[1])
        for idx_ in range(self.spinBox_17.value()):
            idx_using.append(idx_list[sorted_time[idx_][0]])
        # 判定是否会在高负荷期内达到预防周期，是则提前进行预防维修，否则不用操作
        for idx_u in idx_using:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
            if dic[idx_u].loc[dic[idx_u].index[-1], 't1'] + tmp_count['h_time'] >= tp:
                x = dic[idx_u].index[-1]
                dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1']
                dic[idx_u].loc[x + 1, 't1'] = 0  # 因为进行了pm，所以下一阶段的t1一定是0
                dic[idx_u].loc[x + 1, 'life'] = tmp_count[idx_u]['life'].pop()
                dic[idx_u].loc[x + 1, 'state'] = 1
                dic[idx_u].loc[x, 'n_pm'] = 1
                dic[idx_u].loc[x, 'n_cm'] = 0
                dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
                dic[idx_u].loc[x, 'c_cm'] = 0
                dic[idx_u].loc[x, 'c_stop'] = 0
                dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
                dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_pm']
        return dic

    def press_high(self, dic, tmp_count):
        """
        设备在高负荷期运行的模式，国产设备发生故障，产生产能损失，修复完成后，立即投入使用
        进口设备发生故障，由备件接替运行
        params：
            dic:记录信息的字典
            tmp_count:设备固有数据，不会变
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]
        t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
        dtf = min(t_f)
        idx_fault = []
        for idx_ in range(len(idx_using)):
            if t_f[idx_] == dtf:
                idx_fault.append(idx_using[idx_])
        dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
        # 不管怎么样，一下这些操作，两种方案应该都一样的
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 1  # 先进入停机状态，然后再令0号机开始运行
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_f in idx_fault:
            x = dic[idx_f].index[-1]  # 这个时候x已经自加了
            dic[idx_f].loc[x, 't1'] = 0
            dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
            dic[idx_f].loc[x - 1, 'n_cm'] = 1
            dic[idx_f].loc[x - 1, 'c_cm'] = cost_cm(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'c_stop'] = 0
            if idx_f == 0:
                dic[idx_f].loc[x - 1, 'c_stop'] = cost_stop(idx_f, tmp_count)
            dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_stop'] + dic[idx_f].loc[x - 1, 'c_cm']
        if idx_fault[0] not in list(range(self.num_home_press)):
            for idx_u in range(self.num_home_press, self.num_home_press+self.num_import_press):
                if idx_u in idx_fault:
                    dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 3
                else:
                    if dic[idx_u].loc[dic[idx_u].index[-1], 'state'] == 1:
                        continue
                    dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        return dic

    def press_h2l(self, dic, tmp_count, t_next_end):
        """
        设备从高负荷期向低负荷期切换的策略，仅考虑国产压缩机继续运行的情况
        params
            dic:记录设备运行信息
            tmp_count:设备不变的信息
        return:
            dic
        """
        idx_using = [i for i in range(len(dic) - 1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
        dt = t_next_end - dic['t_now_list'][-1]
        dic['t_now_list'].append(t_next_end)
        for idx_u in idx_using:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
            dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
            dic[idx_u].loc[x + 1, 'state'] = 3
            dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']
            dic[idx_u].loc[x, 'n_pm'] = 0
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = 0
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = 0

        for idx_ in range(self.num_home_press):
            dic[idx_].loc[dic[idx_].index[-1], 'state'] = 1
        return dic

    def press_count(self):
        """
        高压泵主题计算过程，包括读入数据，得到最后的计算结果
        :return:
        """
        self.press_params()
        for tp in tqdm(self.range1):
            for _ in range(self.times):
                df_ = pd.DataFrame([[0] * len(self.col_list)], columns=self.col_list)
                dic = {'t_now_list': [0]}
                """
                state 表示状态，1：运行，2：维修，3：待命
                life 表示设备的寿命
                t1表示设备开始运行的的时间
                t2表示设备结束运行的时间
                """

                for idx_machine in range(self.num_high_press + self.num_low_press):
                    dic[idx_machine] = df_.copy()
                    dic[idx_machine].loc[dic[idx_machine].index[-1], 'life'] = self.press_tmp[idx_machine]['life'].pop()
                for idx in range(self.num_low_press):
                    dic[idx]['state'] = 1

                for idx_t in range(len(self.t_table_press)):
                    t_next_start = self.t_table_press.iloc[idx_t, 0]
                    flag, dt = self.press_init_proc(dic, tp)
                    # 低负荷期运行
                    while dic['t_now_list'][-1] + dt < t_next_start:
                        if flag == 1:
                            dic = self.press_low_fault(dic, self.press_tmp)
                        else:
                            dic = self.press_low_pm(dic, tp, self.press_tmp)
                        flag, dt = self.press_init_proc(dic, tp)
                    dic = self.press_l2h(dic, tp, self.press_tmp, t_next_start)
                    t_next_end = self.t_table_press.iloc[idx_t, 1]
                    dt = self.press_init_high(dic, self.press_tmp)  # 只需要考虑哪个会先到平均寿命节点
                    while dic['t_now_list'][-1] + dt < t_next_end:
                        dic = self.press_high(dic, self.press_tmp)
                        dt = self.press_init_high(dic, self.press_tmp)
                    dic = self.press_h2l(dic, self.press_tmp, t_next_end)

                for idx_machine in range(len(dic) - 1):
                    self.all_record_press[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
                    self.all_record_press[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
                    self.all_record_press[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
                    self.all_record_press[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['c_cm'])
                    self.all_record_press[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
                    self.all_record_press[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])

        for idx_machine in range(len(dic) - 1):
            self.all_record_press[idx_machine]['n_stop'] = self.all_record_press[idx_machine]['c_stop'] / \
                                                     self.press_tmp[idx_machine]['cs']



if __name__ == '__main__':
    cgitb.enable(format="text")
    app = QApplication(sys.argv)
    md = demo()
    md.show()
    sys.exit(app.exec_())