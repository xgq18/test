# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:00:38 2020

@author: 65404
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# from press_func import *
def smooth(a, wsz):
    out0 = np.convolve(a, np.ones(wsz, dtype=int), mode='valid') / wsz
    r = np.arange(1, wsz - 1, 2)  # 小于滑动窗口的次级窗口
    start = np.cumsum(a[:wsz - 1])[::2] / r
    stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop)).tolist()

def sampling(Rf, lmda, k_wbl):  # 抽取随机数函数
    tf = (-math.log(1 - Rf)) ** (1 / k_wbl) / lmda
    return tf

def n_cm_func(t1, t2, life):
    """
    t1:初始时间
    t2:结束时间
    life:平均寿命
    该函数是用来计算在t1-t2这个时间段内，发生故障的次数
    """
    return (t2**3-t1**3)/life**3

def mean_life(t):
    """
    beta=3
    平均可用度为0.95
    """
    # T = t/0.2**(1/3)
    """
    beta=3
    平均可用度为0.99时
    """
    T = t/0.04**(1/3)
    return T



def cost_pm(idx_pm, tmp_count):
    """
    预防维修费用为定值
    n_pm:预防维修发生的次数
    tmp_count:用以记录常数的字典
    该函数是用来计算在时间段内预防维修费用
    """
    return tmp_count[idx_pm]['cf']

def cost_cm(idx_fault, tmp_count):
    """
    tmp_count:用以记录常数的字典
    该函数是用来计算在时间段内修复维修所需费用
    """
    return tmp_count[idx_fault]['cmr']

def cost_stop(n_cm, idx, tmp_count):
    """
    time：停机造成损失的时间
    tmp_count:用以记录常数的字典
    该函数是用来计算时间段内停机花费，预防维修也有可能造成停机损失
    """
    return tmp_count[idx]['cs'] * n_cm
    
def Avai(t2, life):
    """
    t2：运行结束的时间
    life:平均寿命
    该函数是用来计算时间段内的平均可用度
    """
    A = 1-(t2/life)**3/4
    return A if A >=0 else 0

def initial_process(dic, tp):
    """
    dic:记录所有的运行信息
    tp： 维修周期
    return flag， dt
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_p = [tp - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtp = min(t_p)
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)
    if dtp > dtf:
        flag = 1
        dt = dtf
    else:
        flag = 2
        dt = dtp
        
    return flag, dt

def initial_high(dic, tp):
    """
    dic:记录所有的运行信息
    tp： 维修周期
    return 
        dt 下一次故障间隔
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
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
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    t_f = [dic[i].loc[dic[i].index[-1], 'life'] - dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtf = min(t_f)
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
    idx_fault = idx_using[t_f.index(dtf)]
    dic['s_now_list'].loc[dic['s_now_list'].index[-1]+1,'number'] = idx_fault
    dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0
    #不管怎么样，一下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 3 #先进入停机状态，
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'A'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
    x = dic[idx_fault].index[-1] #这个时候x已经自加了,df自动更新的
    dic[idx_fault].loc[x, 't1'] = 0 
    dic[idx_fault].loc[x-1, 'state'] = 2
    dic[idx_fault].loc[x, 'life'] = tmp_count[idx_fault]['life'].pop()
    dic[idx_fault].loc[x-1, 'n_cm'] = 1
    dic[idx_fault].loc[x-1, 'c_cm'] = cost_cm(idx_fault, tmp_count)
    dic[idx_fault].loc[x-1, 'A'] = Avai(dic[idx_fault].loc[x-1, 't2'], tmp_count[idx_fault]['mean_life'])
    dic[idx_fault].loc[x-1, 'sum_cost'] = dic[idx_fault].loc[x-1, 'c_cm'] 
    
    #方案一，让国产机器接替运行
    if low_change == 1:
        #国产机器的状态要改为1
        dic[0].loc[dic[0].index[-1], 'state'] = 1 
    else:
    # 方案二，交替运行
        if idx_fault == 0:
            for idx_import in range(1,3,1):
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
    t_p = [tp- dic[i].loc[dic[i].index[-1], 't1'] for i in idx_using]
    dtp = min(t_p)
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtp)
    idx_pm = [idx_using[t_p.index(dtp)]]
    if len(t_p) == 2 and t_p[0] == t_p[-1]:
        idx_pm = [1, 2]
    dic['s_now_list'].loc[dic['s_now_list'].index[-1]+1,'number'] = idx_pm[0]
    dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0 
    #不管怎么样，一下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtp
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 3 #先进入停机状态，然后再令0号机开始运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'A'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
        
    for idx_ in idx_pm:
        x = dic[idx_].index[-1] #这个时候x已经自加了
        dic[idx_].loc[x, 't1'] = 0 
        dic[idx_].loc[x-1, 'state'] = 3
        dic[idx_].loc[x, 'life'] = tmp_count[idx_]['life'].pop()
        dic[idx_].loc[x-1, 'n_pm'] = 1
        dic[idx_].loc[x-1, 'c_pm'] = cost_pm(idx_, tmp_count)
        dic[idx_].loc[x-1, 'A'] = Avai(dic[idx_].loc[x-1, 't2'], tmp_count[idx_]['mean_life'])
        dic[idx_].loc[x-1, 'sum_cost'] = dic[idx_].loc[x-1, 'c_pm']
    
    #方案一，让国产机器接替运行
    if low_change == 1:
         #国产机器的状态要改为1
        dic[0].loc[dic[0].index[-1], 'state'] = 1
    else:
    # 方案二，交替运行
        if int(idx_pm[0]) == 0:
            for idx_import in range(1,3,1):
                dic[idx_import].loc[dic[idx_import].index[-1], 'state'] = 1
        else:
            dic[0].loc[dic[0].index[-1], 'state'] = 1
    return dic

def check_func(dic,tmp_count,tp, t_now,t_next_end):
    """
    dic:可变字典，记录运行轨迹
    tmp_count:不变字典，记录常数
    tp:维修周期
    t_now:现在工作时间
    t_next_end:下次高负荷期结束的时间
    return：
        dic
        
    这个函数是在高负荷期前检查设备是否会在高负荷期前达到维修周期
    若达到则进行pm
    否则不做任何操作
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    dic['t_now_list'].append(t_now)
    dt = dic['t_now_list'][-1] - dic['t_now_list'][-2]
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt #应该是都更新以后才决定action
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 1 #check程序不改变运行状态
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'A'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
        
    dic['s_now_list'].loc[dic['s_now_list'].index[-1]+1,'number'] = -1 
    #占位，并且已经更新，下面的只是修改这个参数而已，也有可能不修改
    dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0
    for idx_u in idx_using:
        x = dic[idx_u].index[-1] 
        if dic[idx_u].loc[x - 1, 't2'] > dic[idx_u].loc[x - 1, 'life']:
            dic[idx_u].loc[x, 't1'] = 0 
            dic[idx_u].loc[x-1, 'state'] = 2
            dic[idx_u].loc[x, 'life'] = tmp_count[idx_u]['life'].pop()
            dic[idx_u].loc[x-1, 'n_cm'] = 1
            dic[idx_u].loc[x-1, 'c_cm'] = cost_cm(idx_u, tmp_count)
            dic[idx_u].loc[x-1, 'A'] = Avai(dic[idx_u].loc[x-1, 't2'], tmp_count[idx_u]['mean_life'])
            dic[idx_u].loc[x-1, 'sum_cost'] = dic[idx_u].loc[x-1, 'c_cm']
            dic['s_now_list'].loc[dic['s_now_list'].index[-1],'number'] = idx_u
            dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0
        elif dic[idx_u].loc[x - 1, 't2'] > tp or dic[idx_u].loc[x - 1, 't2'] + t_next_end - t_now > tp:
            dic[idx_u].loc[x, 't1'] = 0 
            dic[idx_u].loc[x-1, 'state'] = 3
            dic[idx_u].loc[x, 'life'] = tmp_count[idx_u]['life'].pop()
            dic[idx_u].loc[x-1, 'n_pm'] = 1
            dic[idx_u].loc[x-1, 'c_pm'] = cost_pm(idx_u, tmp_count)
            dic[idx_u].loc[x-1, 'A'] = Avai(dic[idx_u].loc[x-1, 't2'], tmp_count[idx_u]['mean_life'])
            dic[idx_u].loc[x-1, 'sum_cost'] = dic[idx_u].loc[x-1, 'c_pm']
            dic['s_now_list'].loc[dic['s_now_list'].index[-1],'number'] = idx_u
            dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0
        
            
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
    dic['s_now_list'].loc[dic['s_now_list'].index[-1]+1,'number'] = -1 #占位，因为t_now_list更新了
    dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0 
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 3 #先进入停机状态，然后再确定哪些机器需要继续运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'A'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
        
    #找出在高负荷期运行的设备编号
    idx_using = [0]
    if dic[1].loc[dic[1].index[-1], 't1'] < dic[2].loc[dic[2].index[-1], 't1']:
        idx_using.append(1)
    else:
        idx_using.append(2)
    
    #判定是否会在高负荷期内达到预防周期，是则提前进行预防维修，否则不用操作
    high_time = 120  #扩展的话，这个也需要重新写
    for idx_u in idx_using:
        dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1
        tmp_ = 0 
        if dic[idx_u].loc[dic[idx_u].index[-1], 't1'] + high_time >= tp:
            tmp_ += 1
            x = dic[idx_u].index[-1]
            dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] 
            dic[idx_u].loc[x+1, 't1'] = 0  #因为进行了pm，所以下一阶段的t1一定是0
            dic[idx_u].loc[x+1, 'state'] = 1 
            dic[idx_u].loc[x+1, 'life'] = tmp_count[idx_u]['life'].pop() 
            dic[idx_u].loc[x, 'n_pm']  = 1
            dic[idx_u].loc[x, 'n_cm'] = 0
            dic[idx_u].loc[x, 'c_pm'] = cost_pm(idx_u, tmp_count)
            dic[idx_u].loc[x, 'c_cm'] = 0
            dic[idx_u].loc[x, 'c_stop'] = 0
            dic[idx_u].loc[x, 'A'] = Avai(dic[idx_u].loc[x, 't2'], tmp_count[idx_u]['mean_life'])
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
    dic['t_now_list'].append(dic['t_now_list'][-1] + dtf)
    idx_fault = idx_using[t_f.index(dtf)]
    dic['s_now_list'].loc[dic['s_now_list'].index[-1]+1,'number'] = idx_fault
    dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0 
    
    #不管怎么样，一下这些操作，两种方案应该都一样的
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dtf
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 3 #先进入停机状态，然后再令0号机开始运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'A'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
    x = dic[idx_fault].index[-1] #这个时候x已经自加了
    dic[idx_fault].loc[x, 't1'] = 0 
    dic[idx_fault].loc[x-1, 'state'] = 2
    dic[idx_fault].loc[x, 'life'] = tmp_count[idx_fault]['life'].pop()
    dic[idx_fault].loc[x-1, 'n_cm'] = 1
    dic[idx_fault].loc[x-1, 'c_cm'] = cost_cm(idx_fault, tmp_count)
       

    if idx_fault == 0:
         dic[idx_fault].loc[x-1, 'c_stop'] = cost_stop(1, idx_fault, tmp_count)
    
    
    dic[idx_fault].loc[x-1, 'A'] = Avai(dic[idx_fault].loc[x-1, 't2'], tmp_count[idx_fault]['mean_life'])
    dic[idx_fault].loc[x-1, 'sum_cost'] = dic[idx_fault].loc[x-1, 'c_cm'] + dic[idx_fault].loc[x-1, 'c_stop']
    
    if idx_fault == 0:
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
        该函数是用设备从高负荷期向低负荷期转换的函数
    return:
        dic
    """
    idx_using = [i for i in range(3) if dic[i].loc[dic[i].index[-1], 'state'] == 1]  # 处于运行状态的设备的编号
    dt = t_next_end - dic['t_now_list'][-1]
    dic['t_now_list'].append(t_next_end) 
    dic['s_now_list'].loc[dic['s_now_list'].index[-1]+1,'number'] = -1 #占位，因为t_now_list更新了
    dic['s_now_list'].loc[dic['s_now_list'].index[-1],'time'] = 0 
    for idx_u in idx_using:
        x = dic[idx_u].index[-1]
        dic[idx_u].loc[x, 't2'] = dic[idx_u].loc[x, 't1'] + dt
        dic[idx_u].loc[x+1, 't1'] = dic[idx_u].loc[x, 't2'] #先假定下一个时期的t1，他们都是上一个时期的t2，然后将fault的改为0
        dic[idx_u].loc[x+1, 'state'] = 3 #先进入停机状态，然后再令0号机开始运行
        dic[idx_u].loc[x+1, 'life'] = dic[idx_u].loc[x, 'life'] #先假定不变，然后再改正
        dic[idx_u].loc[x, 'n_pm']  = 0
        dic[idx_u].loc[x, 'n_cm'] = 0
        dic[idx_u].loc[x, 'c_pm'] = 0
        dic[idx_u].loc[x, 'c_cm'] = 0
        dic[idx_u].loc[x, 'c_stop'] = 0 
        dic[idx_u].loc[x, 'A'] = 0
        dic[idx_u].loc[x, 'now'] = dic['t_now_list'][-1]
        dic[idx_u].loc[x, 'sum_cost'] = 0 
        
    if high_change == 1:
        dic[0].loc[dic[0].index[-1], 'state'] = 1
    else:
        for idx_u in range(1,3,1):
            dic[idx_u].loc[dic[idx_u].index[-1], 'state'] = 1        
    return dic


    


mean_home = mean_life(333)
mean_import = mean_life(333)
rand_list = np.random.rand(300000) #取100w个随机数
k_wbl = 3
life1 = [sampling(i, 1/mean_home, k_wbl) for i in rand_list]
life2 = [sampling(i, 1/mean_import, k_wbl) for i in rand_list]
max_s = 2
min_s = 1
low_change = 1 #一直让国产压缩机运行
high_change = 1 #高负荷期结束的时候，让国产压缩机继续运行

#如果要做扩展，就需要对每个参数单独输入


dic1_import = {'cf':10, #pm的固定费用
             'cv':0.01, #单位时间的pm的可变费用
             'cmr':30, #cm的费用，进口的设备相对来说修理费用低
             't_pm':10,
             't_cm':15, #cm需要的时间
             'cs':50, #单位时间的产能损失
             'r':0, #3%的年折现率，对应就是8e-5的天折现率
             'mean_life':mean_import,
             'life':life2}

dic2_import = dic1_import
dic2_import['life'] = dic1_import['life'][::-1]

# life1 = [2000,2000, 2000, 10, 10, 10, 10]
dic_home = {'cf':30, #pm的固定费用 
             'cv':0.02, #单位时间的pm的可变费用
             'cmr':75, #cm的费用
             't_pm':10,
             't_cm':15, #cm需要的时间
             'cs':100, #单位时间的产能损失   
             'r':0, #r月份折现率
             'mean_life':mean_home,
             'life':life1}

tmp_count = {0:dic_home,
             1:dic1_import,
             2:dic2_import,}



initial_time = 240
th_time = 120
year = 5
number_machine = 3
th_start_list = [initial_time + i * 360 for i in range(year)]
th_end_list = [(i + 1) * 360 for i in range(year)]
t_table = pd.DataFrame()
t_table['th_start'] = th_start_list
t_table['th_end'] = th_end_list


cols=['n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'A', 'sum_cost']
range1 = range(800, 1001, 10)
idxs = [str(tp) for tp in range1]
df = pd.DataFrame(np.zeros((len(idxs), len(cols))), index=idxs, columns=cols)
all_record = {0:df.copy(),
              1:df.copy(),
              2:df.copy()}

dic = {0:pd.DataFrame([[0]*12], columns=['t1','t2','state','life','n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'A', 'now','sum_cost']),
        1:pd.DataFrame([[0]*12], columns=['t1','t2','state','life','n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'A', 'now','sum_cost']),
        2:pd.DataFrame([[0]*12], columns=['t1','t2','state','life','n_pm', 'n_cm', 'c_pm', 'c_cm', 'c_stop', 'A', 'now','sum_cost']),
        'inventory':2,
        'c_holding':0,
        't_now_list':[0], #设备现在运行时间
        's_now_list':pd.DataFrame(np.zeros((1, 2),dtype='int'), columns=['number', 'time'])} #设备发生故障的设备的编号
"""
state 表示状态，1：运行，2：维修，3：待命
life 表示设备的寿命
t1表示设备开始运行的的时间
t2表示设备结束运行的时间
"""
all_record = {0:df.copy(),
              1:df.copy(),
              2:df.copy()}
times = 10
for tp in range1:
    for _ in range(times):
        for idx_machine in range(3):
            dic[idx_machine].loc[dic[0].index[-1], 'life'] = tmp_count[idx_machine]['life'].pop() #给设备加上寿命
        dic[0].loc[dic[0].index[-1], 'state'] = 1 #每次都是让国产压缩机开始运行
        dic[1].loc[dic[1].index[-1], 'state'] = 3
        dic[2].loc[dic[2].index[-1], 'state'] = 3
            
        for idx_t in range(len(t_table)):
            t_next_start = t_table.iloc[idx_t, 0]
            flag, dt = initial_process(dic, tp)
            """
            flag：1，表示下一次是故障时间
            flag：2，表示下一次是预防维修
            """
            #低负荷期运行
            while dic['t_now_list'][-1] + dt  < t_next_start:
                if flag == 1:
                    dic = low_fault(dic, tmp_count, low_change)
                else:
                    dic = low_pm(dic, tp, tmp_count, low_change)
                flag, dt = initial_process(dic, tp)  
                
            t_next_end = t_table.iloc[idx_t, 1]
            dic = low2high(dic, tp, tmp_count, t_next_start)    
            dt = initial_high(dic, tp)
            while dic['t_now_list'][-1] + dt  < t_next_end:
                dic = high_fault(dic, tmp_count)
                dt = initial_high(dic, tp) 
            dic = high2low(dic, tmp_count, t_next_end, high_change)
            
        for idx_machine in range(3):
           all_record[idx_machine].loc[str(tp), 'n_pm'] += np.nansum(dic[idx_machine]['n_pm'])
           all_record[idx_machine].loc[str(tp), 'n_cm'] += np.nansum(dic[idx_machine]['n_cm'])
           all_record[idx_machine].loc[str(tp), 'c_pm'] += np.nansum(dic[idx_machine]['c_pm'])
           all_record[idx_machine].loc[str(tp), 'c_cm'] += np.nansum(dic[idx_machine]['c_cm'])
           all_record[idx_machine].loc[str(tp), 'c_stop'] += np.nansum(dic[idx_machine]['c_stop'])
           all_record[idx_machine].loc[str(tp), 'A'] += np.nanmean(dic[idx_machine]['A'])
           all_record[idx_machine].loc[str(tp), 'sum_cost'] += np.nansum(dic[idx_machine]['sum_cost'])      

dic2 = {}
for i in range(3):
    dic2[i] = dic[i]   

x_num = [int(x) for x in all_record[0].index]
# y_num = smooth((all_record[0]['sum_cost']+all_record[1]['sum_cost']+all_record[2]['sum_cost'])/times, 5)
y_num = (all_record[0]['sum_cost']+all_record[1]['sum_cost']+all_record[2]['sum_cost'])/times
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(x_num, y_num)
plt.title('平均花费随维修周期变化的示意图,12')
plt.xlabel('维修周期/d')
plt.ylabel('花费/万元')
plt.show()



# lis = (all_record[0]['sum_cost']+all_record[1]['sum_cost']+all_record[2]['sum_cost'])/times
# idx = lis.index.tolist()
# lis = y_num
# d = idx[lis.index(min(lis))]
# fare = min(lis)


