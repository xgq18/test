# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 16:01:47 2020
仿真程序，在理论的基础上加以改进
@author: 65404
测试一下如何git自己上传
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time


def sampling(Rf, lmda, k_wbl):  # 抽取随机数函数
    tf = (-math.log(1 - Rf)) ** (1 / k_wbl) / lmda
    return tf


def mean_life(t):
    """
    beta=3
    平均可用度为0.99时
    """
    T = t / 0.04 ** (1 / 3)
    return T


def cost_pm(idx_pm, tmp_count):
    """
    idx_pm:发生预防维修设备的编号
    tmp_count:用以记录常数的字典
    该函数是用来计算在时间段内预防维修费用
    """
    return tmp_count[idx_pm]['cf']


def cost_cm(idx_fault, tmp_count):
    """
    idx_fault:发生故障的设备的编号
    tmp_count:用以记录常数的字典
    该函数是用来计算在时间段内修复维修所需费用
    """
    return tmp_count[idx_fault]['cmr']


def cost_stop(idx, tmp_count):
    """
    idx：发生故障停机的设备编号
    tmp_count:用以记录常数的字典
    该函数是用来计算时间段内停机花费，预防维修也有可能造成停机损失
    """
    return tmp_count[idx]['cs']


def initial_process(dic, tp):
    """
    dic:记录所有的运行信息
    tp： 维修周期
    return flag， dt
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
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


def initial_high(dic, tmp_count):
    """
    dic:记录所有的运行信息
    tp： 维修周期
    return
        dt 下一次故障间隔
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)
    return dtf


def low_fault(dic, tmp_count, low_change):
    """
    低负荷发生故障的处理程序
    dic:记录信息的字典,改变
    tmp_count:设备固有数据，不会变
    low_change: 低负荷期的切换策略
    return:
        dic
    """

    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
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
        dic[idx_u].loc[x + 1, 'state'] = 3  # 先进入停机状态，
        dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
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

    if low_change == 1:
        dic[0].loc[dic[0].index[-1], 'state'] = 1
    else:
        if 0 in idx_fault:
            for idx_import in range(1, 3, 1):
                dic[idx_import].loc[dic[idx_import].index[-1], 'state'] = 1
        else:
            dic[0].loc[dic[0].index[-1], 'state'] = 1
    return dic


def low_pm(dic, tp, tmp_count, low_change):
    """
    dic:记录信息的字典
    tp:维修周期
    tmp_count:设备固有数据，不会变
    low_change: 低负荷期的切换策略
    需要考虑两台同时维修
    return:
        dic
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtp = min(t_p)
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
    idx_pm = [idx_using[t_p.index(dtp)]]
    if len(t_p) == 2 and t_p[0] == t_p[-1]:
        idx_pm = [1, 2]  # 考虑两台一起维修
    # 不管怎么样，一下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
        dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x + 1, 'state'] = 3  # 先进入停机状态，然后再令0号机开始运行
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
        dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
        dic[idx_].loc[x - 1, 'n_pm'] = 1
        dic[idx_].loc[x - 1, 'c_pm'] = cost_pm(idx_, tmp_count)
        dic[idx_].loc[x - 1, 'sum_cost'] = dic[idx_].loc[x - 1, 'c_pm']

    # 方案一，让国产机器接替运行
    if low_change == 1:
        # 国产机器的状态要改为1
        dic[0].loc[dic[0].index[-1], 'state'] = 1
    else:
        # 方案二，交替运行
        if int(idx_pm[0]) == 0:
            for idx_import in range(1, 3, 1):
                dic[idx_import].loc[dic[idx_import].index[-1], 'state'] = 1
        else:
            dic[0].loc[dic[0].index[-1], 'state'] = 1
    return dic


def low2high(dic, tp, tmp_count, t_next_start):
    """
    这个过程不考虑停机状态的更新了。
    dic:记录信息的字典,改变
    tmp_count:设备固有数据，不会变
    t_next_start: 高负荷期开始的时间
    tp:维修周期
    return:
        dic
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
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

    # 找出在高负荷期运行的设备编号
    idx_using = [0]
    if dic[1].loc[dic[1].index[-1], 't1'] < dic[2].loc[dic[2].index[-1], 't1']:
        idx_using.append(1)
    else:
        idx_using.append(2)

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
            dic[idx_u].loc[x, 'n_cm'] = 0  # 假设一定是在低负荷期开始运行
            dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_pm']

    return dic


def high_fault(dic, tmp_count):
    """
    dic:记录信息的字典
    tmp_count:设备固有数据，不会变
    return:
        dic
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)
    idx_fault = [idx_using[t_f.index(dtf)]]
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
    # 不管怎么样，一下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
        dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x + 1, 'state'] = 3  # 先进入停机状态，然后再令0号机开始运行
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
        if idx_f == 0:
            dic[idx_f].loc[x - 1, 'c_stop'] = cost_stop(idx_f, tmp_count)
        dic[idx_f].loc[x - 1, 'sum_cost'] = dic[idx_f].loc[x - 1, 'c_stop'] + dic[idx_f].loc[x - 1, 'c_cm']

    if 0 in idx_fault:
        for idx_u in idx_using:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    else:
        idx_tmp = [0]
        for idx_u in range(1, 3, 1):
            if idx_u not in idx_using:
                idx_tmp.append(idx_u)
        for idx_u in idx_tmp:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1

    return dic


def high2low(dic, tmp_count, t_next_end, high_change):
    """
    dic:记录设备运行信息
    tmp_count:设备不变的信息
    t_next_end:高负荷结束的时间
    high_change：高负荷期结束时的切换策略
        该函数是用设备从高负荷期向低负荷期转换的函数
    return:
        dic
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    dt = t_next_end - dic['t_now_list'][-1]
    dic['t_now_list'].append(t_next_end)
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
        dic[idx_u].loc[x + 1, 't1'] = dic[idx_u].loc[x, 't2']  # 先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x + 1, 'state'] = 3  # 先进入停机状态，然后再令0号机开始运行
        dic[idx_u].loc[x + 1, 'life'] = dic[idx_u].loc[x, 'life']  # 先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm'] = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0
    if high_change == 1:
        dic[0].loc[dic[0].index[-1], 'state'] = 1
    else:
        for idx_u in range(1, 3, 1):
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    return dic


# 假设设计寿命一样
mean_home = mean_life(333)
mean_import = mean_life(333)
np.random.seed(1)
rand_list = np.random.rand(15000)  # 随机数
k_wbl = 3
life1 = [sampling(i, 1 / mean_home, k_wbl) for i in rand_list]
life2 = [sampling(i, 1 / mean_import, k_wbl) for i in rand_list]
low_change = 2  # 一直让国产压缩机运行
high_change = 1  # 高负荷期结束的时候，让国产压缩机继续运行

# 如果要做扩展，就需要对每个参数单独输入


dic1_import = {'cf': 10,  # pm的固定费用
               'cmr': 30,  # cm的费用，进口的设备相对来说修理费用低
               'cs': 3000,  # 单次损失
               'mean_life': mean_import,
               'life': life2}

dic2_import = dic1_import
dic2_import['life'] = dic1_import['life'][::-1]

dic_home = {'cf': 30,  # pm的固定费用
            'cmr': 75,  # cm的费用
            'cs': 6000,  # 单位时间的产能损失
            'mean_life': mean_home,
            'life': life1}
th_time = 120
tmp_count = {0: dic_home,
             1: dic1_import,
             2: dic2_import,
             'h_time': th_time}

initial_time = 240
year = 25
d_year = 360
th_start_list = [initial_time + i * d_year for i in range(year)]
th_end_list = [(i + 1) * d_year for i in range(year)]
t_table = pd.DataFrame()
t_table['th_start'] = th_start_list
t_table['th_end'] = th_end_list

cols = ['n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'sum_cost']
range1 = range(200, 501, 10)
idxs = [str(tp) for tp in range1]
df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
all_record = {0: df.copy(),
              1: df.copy(),
              2: df.copy()}
times = 100
col_list = ['t1', 't2', 'state', 'life', 'n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'now', 'sum_cost']

a = time.time()

for tp in tqdm(range1):
    for _ in range(times):
        dic = {0: pd.DataFrame([[0] * len(col_list)], columns=col_list),
               1: pd.DataFrame([[0] * len(col_list)], columns=col_list),
               2: pd.DataFrame([[0] * len(col_list)], columns=col_list),
               't_now_list': [0]}  # 设备现在运行时间
        """
        state 表示状态，1：运行，2：维修，3：待命
        life 表示设备的寿命
        t1表示设备开始运行的的时间
        t2表示设备结束运行的时间
        """

        for idx_machine in range(3):
            dic[idx_machine].loc[dic[0].index[-1], 'life'] = tmp_count[idx_machine]['life'].pop()  # 给设备加上寿命
        dic[0].loc[dic[0].index[-1], 'state'] = 1  # 每次都是让国产压缩机开始运行
        dic[1].loc[dic[1].index[-1], 'state'] = 3
        dic[2].loc[dic[2].index[-1], 'state'] = 3

        for idx_t in range(len(t_table)):
            t_next_start = t_table.iloc[idx_t, 0]
            flag, dt = initial_process(dic, tp)

            # 低负荷期运行
            while dic['t_now_list'][-1] + dt < t_next_start:
                if flag == 1:
                    dic = low_fault(dic, tmp_count, low_change)
                else:
                    dic = low_pm(dic, tp, tmp_count, low_change)
                flag, dt = initial_process(dic, tp)

            dic = low2high(dic, tp, tmp_count, t_next_start)
            t_next_end = t_table.iloc[idx_t, 1]
            dt = initial_high(dic, tmp_count)  # 只需要考虑哪个会先到平均寿命节点
            while dic['t_now_list'][-1] + dt < t_next_end:
                dic = high_fault(dic, tmp_count)
                dt = initial_high(dic, tmp_count)
            dic = high2low(dic, tmp_count, t_next_end, high_change)

        for idx_machine in range(3):
            all_record[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
            all_record[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
            all_record[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
            all_record[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['c_cm'])
            all_record[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
            all_record[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])

b = time.time()
print("花费时间", b - a)
dic2 = {}
for i in range(3):
    dic2[i] = dic[i]

times = 1
x_num = [int(x) for x in all_record[0].index]
y_num = (all_record[0]['sum_cost'] + all_record[1]['sum_cost'] + all_record[2]['sum_cost']) / times
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(x_num, y_num)
plt.title('多台设备25年费pm_up')
plt.xlabel('维修周期/d')
plt.ylabel('花费/万元')
plt.show()

# np.save('多台设备25年费pm_up.npy',all_record)

