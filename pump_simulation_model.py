# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:19:19 2020
高压泵的仿真模型 simulation
console3
@author: 65404
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def sampling(Rf, lmda, k_wbl):  # 抽取随机数函数
    tf = (-math.log(1 - Rf)) ** (1 / k_wbl) / lmda
    return tf

def mean_life(t):
    """
    beta=3
    平均可用度为0.99时
    """
    T = t/0.04**(1/3)
    return T

def cost_pm(idx_pm, tmp_count):
    """
    idx_pm：预防维修的设备的编号
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
    idx：发生产能损失的设备
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
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dt = min(t_p)  
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)
    if dt < dtf:
        flag = 2
    else:
        flag = 1
        dt = dtf
    return  flag, dt

def initial_high(dic, tmp_count):
    """
    dic:记录所有的运行信息
    tp： 维修周期
    return 
        dt 下一次故障间隔
    """
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)       
    return  dtf

def low_fault(dic, tmp_count, low_change):
    
    """
    dic:记录信息的字典,改变
    tmp_count:设备固有数据，不会变
    low_change: 低负荷期的切换策略
    return:
        dic
    """
    
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
    idx_fault = []
    for idx_ in range(len(idx_using)):
        if t_f[idx_] == dtf:
            idx_fault.append(idx_using[idx_])
    #不管怎么样，以下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 1 
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life']
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
    for idx_f in idx_fault:
        x = dic[idx_f].index[-1] #这个时候x已经自加了,df自动更新的
        dic[idx_f].loc[x, 't1'] = 0 
        dic[idx_f].loc[x, 'state'] = 3
        dic[idx_f].loc[x, 'life'] = tmp_count[idx_f]['life'].pop()
        dic[idx_f].loc[x-1, 'n_cm'] = 1
        dic[idx_f].loc[x-1, 'c_cm'] = cost_cm(idx_f, tmp_count)
        dic[idx_f].loc[x-1, 'sum_cost'] = dic[idx_f].loc[x-1, 'c_cm'] 
    
    idx_spare = [i for i in range(len(dic)-1) if i not in idx_using]
    n_pm = len(idx_fault)
    if n_pm > 2:
        for idx_u in idx_fault:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    elif n_pm == 2:
        idx_list = [i for i in range(len(dic)-1) if i not in idx_fault]
        for idx_u in idx_list:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    else:
        if low_change == 1:
            idx_u = low_change1(dic, idx_spare)
        elif low_change == 2:
            idx_u = low_change2(dic, idx_spare)
        dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1   
    return dic


def low_change1(dic, idx_spare):
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
    
def low_change2(dic, idx_spare):
    """
    找到运行时间最长的备件来接替运行
    params：
        dic:运行记录的字典
        idx_spare:备件的编号
    return：
        idx
    """
    oper_time = [dic[i].loc[dic[i].index[-1], 't1'] for i in idx_spare]
    idx = idx_spare[oper_time.index(max(oper_time))]
    return idx

def low_pm(dic, tp, tmp_count, low_change):
    
    """
    dic:记录信息的字典
    tp:维修周期
    tmp_count:设备固有数据，不会变
    low_change: 低负荷期的切换策略
    需要考虑多台设备同时维修
    return:
        dic
    """
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_p = [tp- dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtp = min(t_p)
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
    idx_pm = []
    for idx_ in range(len(idx_using)):
        if t_p[idx_] == dtp:
            idx_pm.append(idx_using[idx_])
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 1 
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0
        
    for idx_ in idx_pm:
        x = dic[idx_].index[-1] #这个时候x已经自加了
        dic[idx_].loc[x, 't1'] = 0 
        dic[idx_].loc[x, 'state'] = 3
        dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
        dic[idx_].loc[x-1, 'n_pm'] = 1
        dic[idx_].loc[x-1, 'c_pm'] = cost_pm(idx_, tmp_count)
        dic[idx_].loc[x-1, 'sum_cost'] += dic[idx_].loc[x-1, 'c_pm'] 
    
     # 这个得分情况讨论吧，如果同时维修数大于2，则直接原地维修
    #如果小于2，则可以考虑接替
    idx_spare = [i for i in range(len(dic)-1) if i not in idx_using]
    n_pm = len(idx_pm)
    if n_pm > 2:
        for idx_u in idx_pm:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    elif n_pm == 2:
        idx_using = [i for i in range(len(dic)-1) if i not in idx_pm]
        for idx_u in idx_using:
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    else:

        if low_change == 1:
            idx_u = low_change1(dic, idx_spare)
        elif low_change == 2:
            idx_u = low_change2(dic, idx_spare)
        dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
    return dic

def low2high(dic, tp, tmp_count, t_next_start):
    """
    设备从低负荷期到高负荷期的转换过程
    params:
        dic:记录信息的字典,改变
        tmp_count:设备固有数据，不会变
        t_next_start: 高负荷期开始的时间
        tp:维修周期
    return:
        dic
    """
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    dt = t_next_start - dic['t_now_list'][-1]
    dic['t_now_list'].append(t_next_start) 
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 1 #先进入停机状态，然后再确定哪些机器需要继续运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0
        
    idx_using = range(len(dic)-1)
    #判定是否会在高负荷期内达到预防周期，是则提前进行预防维修，否则不用操作
    for idx_u in idx_using:
        dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        if dic[idx_u].loc[dic[idx_u].index[-1], 't1'] + tmp_count['h_time'] >= tp:
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] 
            dic[idx_u].loc[x+1, 't1'] = 0  #因为进行了pm，所以下一阶段的t1一定是0
            dic[idx_u].loc[x+1, 'life'] = tmp_count[idx_u]['life'].pop()
            dic[idx_u].loc[x+1, 'state'] = 1 
            dic[idx_u].loc[x, 'n_pm']  = 1
            dic[idx_u].loc[x, 'n_cm'] = 0  
            dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
            dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_pm'] 
               
    return dic

def high_fault(dic, tmp_count):
    """
    设备在高负荷期运行的模式
    params：
        dic:记录信息的字典
        tmp_count:设备固有数据，不会变
    return:
        dic
    """
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f) 
    idx_fault = []
    for idx_ in range(len(idx_using)):
        if t_f[idx_] == dtf:
            idx_fault.append(idx_using[idx_])
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)    
    #不管怎么样，一下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 1 #先进入停机状态，然后再令0号机开始运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = dic[idx_u].loc[x, 'c_cm']
        
    for idx_f in idx_fault:
        x = dic[idx_f].index[-1] #这个时候x已经自加了
        dic[idx_f].loc[x, 't1'] = 0 
        dic[idx_f].loc[x-1, 'c_stop'] = cost_stop(idx_f, tmp_count)
        dic[idx_f].loc[x-1, 'sum_cost'] += dic[idx_f].loc[x-1, 'c_stop']   
    return dic

def high_change1(dic, num):
    """
    策略1：停掉运行时间最长的两台高压泵
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic)-1)]
    life_list = np.array(life_list) #必须是数组形式，才能使用负号输出
    sorted_list = np.argsort(-life_list)
    return sorted_list[:num]

def high_change2(dic, num):
    """
    策略2：停掉运行时间最短的num台高压泵
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic)-1)]
    life_list = np.array(life_list) #必须是数组形式，才能使用负号输出
    sorted_list = np.argsort(life_list)
    return sorted_list[:num]

def high_change3(dic, num):
    """
    策略3：停掉运行时间处于中间的num台高压泵
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic)-1)]
    sorted_nums = sorted(enumerate(life_list), key=lambda x: x[1])
    df = pd.DataFrame(sorted_nums, columns=['idx', 'num'])
    num_machine = len(dic)-1
    idx_start = (num_machine-num)//2
    range1 = df['idx'][idx_start: idx_start+num].values
    return range1

def high_change4(dic, num):
    """
    策略4：按顺序停掉其中num台泵
    要求总的设备数是num的整数倍，这样才能形成循环
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    num1 = (len(dic)-1)//num #循环数1，表征经过几次停机形成一次循环
    num2 = dic['t_now_list'][-1] // 360 %num1
    range1 = range(num2 * num, num2*num+num)
    return range1

def high_change5(dic, num):
    """
    策略5：停掉距离最近的num台泵
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    life_list = [dic[i].loc[dic[i].index[-1], 't1'] for i in range(len(dic)-1)]
    sorted_nums = sorted(enumerate(life_list), key=lambda x: x[1])
    df = pd.DataFrame(sorted_nums, columns=['idx', 'num'])
    df['diff']=df['num'].diff() #累减函数，从小往上1个步长
    df['diff'] = df['diff'].fillna(1000000)
    df = df.sort_values(by="diff", axis=0, ascending=True) #axis=0就是表示的一列
    range1 = df['idx'][:num]
    return range1

def high_change6(dic, num):
    """
    策略6：停掉两台进口的高压泵，这个策略是压缩机独有的
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    return [4, 5]

def high_change7(dic, num):
    """
    策略7：依次停掉两台国产的高压泵，这个策略是压缩机独有的
    params:
        dic:记录设备运行信息的字典，可变
        num：需要停掉的设备数目
    return:
        idx：num个需要停机的设备的编号
    """
    num1 = 2 #一定是两次停机形成一个循环
    num2 = dic['t_now_list'][-1] // 360 %num1
    range1 = range(num2 * num, num2*num+num)
    return range1

def high2low(dic, tmp_count, t_next_end, high_change, num):
    """
    设备从高负荷期向低负荷期切换的策略，这个策略比较多
    params
        dic:记录设备运行信息
        tmp_count:设备不变的信息
    return:
        dic
    """
    idx_using = [i for i in range(len(dic)-1) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    dt = t_next_end - dic['t_now_list'][-1]
    dic['t_now_list'].append(t_next_end) 
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 1 #先进入停机状态，然后再令0号机开始运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0  
    if high_change == 1:
        idx_list = high_change1(dic, num)
    elif high_change == 2:
        idx_list = high_change2(dic, num)
    elif high_change == 3:
        idx_list = high_change3(dic, num)
    elif high_change == 4:
        idx_list = high_change4(dic, num)
    elif high_change == 5:
        idx_list = high_change5(dic, num)
    elif high_change == 6:
        idx_list = high_change6(dic, num)
    else:
        idx_list = high_change7(dic, num)
        
    for idx in idx_list:
        dic[idx].loc[dic[idx].index[-1], 'state'] = 3
    return dic


    

#假设设计寿命一样
mean_home = mean_life(333)
mean_import = mean_life(333)
np.random.seed(1)
rand_list = np.random.rand(1500) #随机数
k_wbl = 3
life1 = [sampling(i, 1/mean_home, k_wbl) for i in rand_list]
life2 = [sampling(i, 1/mean_import, k_wbl) for i in rand_list]
low_change = 1
high_change = 1

#如果要做扩展，就需要对每个参数单独输入
num_low = 4
num_high = 2

dic_import = {'cf':10, #pm的固定费用
             'cmr':30, #cm的费用，进口的设备相对来说修理费用低
             'cs':600, #单次损失
             'mean_life':mean_import,
             'life':life2}

dic_home = {'cf':30, #pm的固定费用 
             'cmr':75, #cm的费用
             'cs':600, #单位时间的产能损失   
             'mean_life':mean_home,
             'life':life1}
tmp_count = {}
for idx in range(num_low):
    tmp_count[idx] = dic_home.copy()
    tmp_count[idx]['life'] = life2.copy()[:-(idx+1)*10]
   
for idx in range(num_low, num_low+num_high):
    tmp_count[idx] = dic_import.copy()
    tmp_count[idx]['life'] = life1.copy()[:-(idx+1)*10]



th_time = 120
tmp_count['h_time'] = th_time
initial_time = 240
year = 25
d_year = 360
th_start_list = [initial_time + i * d_year for i in range(year)]
th_end_list = [(i + 1) * d_year for i in range(year)]
t_table = pd.DataFrame()
t_table['th_start'] = th_start_list
t_table['th_end'] = th_end_list


cols=['n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'sum_cost']
range1 = range(200, 1001, 10)
idxs = [str(tp) for tp in range1]
df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
all_record = {}
for idx in range(num_low+num_high):
    all_record[idx] = df.copy()
times = 1
col_list = ['t1','t2','state','life','n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop','now','sum_cost']

for tp in tqdm(range1):
    for _ in range(times):
        df_ = pd.DataFrame([[0]*len(col_list)], columns=col_list)
        dic = {'t_now_list':[0]}      
        """
        state 表示状态，1：运行，2：维修，3：待命
        life 表示设备的寿命
        t1表示设备开始运行的的时间
        t2表示设备结束运行的时间
        """
        
        for idx_machine in range(num_high+num_low):
            dic[idx_machine] = df_.copy()
            dic[idx_machine].loc[dic[idx_machine].index[-1], 'life'] = tmp_count[idx_machine]['life'].pop() 
        for idx in range(num_low):
            dic[idx]['state'] = 1
        
        for idx_t in range(len(t_table)):
            t_next_start = t_table.iloc[idx_t, 0]
            flag, dt = initial_process(dic, tp)
            #低负荷期运行
            while dic['t_now_list'][-1] + dt  < t_next_start:
                if flag == 1:
                    dic = low_fault(dic, tmp_count, low_change)
                else:
                    dic = low_pm(dic, tp, tmp_count, low_change)
                flag, dt = initial_process(dic, tp)   
            dic = low2high(dic, tp, tmp_count, t_next_start)    
            t_next_end = t_table.iloc[idx_t, 1]
            dt = initial_high(dic, tmp_count) #只需要考虑哪个会先到平均寿命节点
            while dic['t_now_list'][-1] + dt  < t_next_end:
                dic = high_fault(dic, tmp_count)
                dt = initial_high(dic, tmp_count) 
            num = 2
            dic = high2low(dic, tmp_count, t_next_end, high_change, num)
            
        for idx_machine in range(len(dic)-1):
                all_record[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
                all_record[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
                all_record[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
                all_record[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['c_cm'])
                all_record[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
                all_record[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])
 


dic2 = {}
for i in range(num_high+num_low):
    dic2[i] = dic[i]       
    


x_num = [int(x) for x in all_record[0].index]
y_num = 0
for i in range(len(dic)-1):
    y_num += all_record[i]['sum_cost']
    
y_num /= times
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(x_num, y_num)
plt.title('高压泵多台设备理论结果1'+str(high_change))
plt.xlabel('维修周期/d')
plt.ylabel('花费/万元')
plt.show()


np.save('高压泵多台设备理论结果1'+str(high_change)+'.npy',all_record)

