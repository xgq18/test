# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:16:11 2019

@author: XGQ18
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:29:35 2019

@author: XGQ18
"""

import numpy as np
import pandas as pd
from math import log
import matplotlib.pyplot as plt
import time
pd.set_option("max_columns", 1000)

#抽样函数
def sampling(i, rand, lmda, k_wbl):  # 抽取随机数函数
    Rf = rand[i]
    tf = (-log(1-Rf))**(1/k_wbl)/lmda
    return tf

operation_number = 6
#低负荷期发生故障的函数
#只要执行这个函数就表示在低负荷期发生了故障
def fault_in_low(t_record, s_record, t_fault_list, t_now_list, th_start, x, y1, y2):
    index_spare = [i for i in range(6) if s_record.iloc[i, -1] != 1]
    t_fault_spare = [t_fault_list[i] for i in index_spare]
    index_using = [i for i in range(6) if s_record.iloc[i, -1] == 1]
    t_fault_temp = t_fault_list.copy()
    for i in index_spare:
        t_fault_temp[i] = 1000000
    dtf = [t_fault_temp[i] - t_record.iloc[i, -1] for i in range(6)]
    dtf_min = min(dtf)
    index_fault = dtf.index(min(dtf))
    t_now_fault = t_now_list[-1] + dtf_min  # 这个表示当前故障发生的时间，并且肯定在高负荷前
    t_now_list.append(t_now_fault)
    if index_fault == 4 or index_fault == 5:
        t_fault_list[index_fault] = life2[y2]
        y2 += 1
    else:
        t_fault_list[index_fault] = life1[y1]
        y1 += 1
    index_change = index_spare[t_fault_spare.index(min(t_fault_spare))]
    s_record_now = [s_record.iloc[i, -1] for i in range(6)]
    s_record_now[index_fault] = 2
    s_record_now[index_change] = 1
    s_record[x + 1] = s_record_now

    t_record_now = [t_record.iloc[i, -1] for i in range(6)]
    for i in index_using:
        t_record_now[i] += dtf_min

    t_record_now[index_fault] = 0
    t_record[x + 1] = t_record_now
    x += 1
    return t_record, s_record, t_fault_list, t_now_list, x, y1, y2


# 低负荷期预防性维修
def pm_in_low(t_record, s_record, t_fault_list, t_now_list, th_start, x, y1, y2):
    index_spare = [i for i in range(6) if s_record.iloc[i, -1] != 1]
    t_operation_spare = [t_record.iloc[i, -1] for i in index_spare]
    index_using = [i for i in range(6) if s_record.iloc[i, -1] == 1]
    dtp = [tp - t_record.iloc[i, -1] for i in range(6)]
    for i in index_spare:
        dtp[i] = 1000000
    dtp_min = min(dtp)
    index_min_pm = dtp.index(min(dtp))
    t_next_pm = t_now_list[-1] + dtp_min
    t_now_list.append(t_next_pm)
    if index_min_pm == 4 or index_min_pm == 5:
        t_fault_list[index_min_pm] = life2[y2]
        y2 += 1
    else:
        t_fault_list[index_min_pm] = life1[y1]
        y1 += 1

    index_change = index_spare[t_operation_spare.index(min(t_operation_spare))]

    s_record_now = [s_record.iloc[i, -1] for i in range(6)]
    s_record_now[index_min_pm] = 3
    s_record_now[index_change] = 1
    s_record[x + 1] = s_record_now
    t_record_now = [t_record.iloc[i, -1] for i in range(6)]
    for i in index_using:
        t_record_now[i] += dtp_min

    t_record_now[index_min_pm] = 0
    t_record[x + 1] = t_record_now
    x += 1
    return t_record, s_record, t_fault_list, t_now_list, x, y1, y2
#低负荷期转向高负荷期函数
#在执行这个函数之前，一定要保证不会再发生PM和故障
def low_to_high(t_record, s_record, t_now_list, th_start, x):
    dt = th_start - t_now_list[-1]
    t_now_list.append(th_start)
    index_spare = [i for i in range(6) if s_record.iloc[i, -1] != 1]
    t_record_now = [t_record.iloc[i, -1] + dt for i in range(6)]
    for i in index_spare:
        t_record_now[i] = t_record.iloc[i, -1]
        
    s_record[x+1] = [1 for i in range(6)]
    t_record[x+1] = t_record_now
    x += 1
    return t_record, s_record, t_now_list, x

#高负荷期故障函数
#执行这个函数就一定要保证下一次的故障一定是发生在高负荷期内
def fault_in_high(t_record, s_record, t_fault_list, t_now_list, th_end, x, y1, y2):
    dtf = [t_fault_list[i] - t_record.iloc[i, -1] for i in range(6)]
    dtf_min = min(dtf)        
    index_fault = dtf.index(min(dtf))
    t_now_fault = t_now_list[-1] + dtf_min
    t_now_list.append(t_now_fault)
    if index_fault == 4 or index_fault == 5:
        t_fault_list[index_fault] = life2[y2]
        y2 += 1
    else:
        t_fault_list[index_fault] = life1[y1]
        y1 += 1

    s_record_now = [1 for i in range(6)]
    s_record_now[index_fault] = 2
    s_record[x+1] = s_record_now
    t_record_now = [t_record.iloc[i, -1] + dtf_min for i in range(6)]
    t_record_now[index_fault] = 0
    t_record[x+1] = t_record_now
    x+=1 
    
    dtf = [t_fault_list[i] - t_record.iloc[i, -1] for i in range(6)]
    dtf_min = min(dtf)
    index_fault = dtf.index(min(dtf))
    t_next_fault = t_now_list[-1] + dtf_min
    return t_record, s_record, t_fault_list, t_now_list,t_next_fault, x, y1, y2 

#这个函数只能寻找当前间隔值最小的索引
def get_index(t_record_now):
    s = t_record_now.copy()
    s.sort()
    list_temp = []
    for i in range(len(s)-1):
        list_temp.append(s[i+1] - s[i]) 
    min_gap = min(list_temp)
    for i in range(len(s)-1):
        gap = s[i+1] - s[i]
        if gap == min_gap:
            min_index = i
    for i in range(len(t_record_now)):
        if t_record_now[i] == s[min_index]:
            delete_index = i   
            break
    return delete_index

#高负荷期转向低负荷期的函数
#执行这个函数之前一定要保证不会再发生故障
def high_to_low(t_record, s_record, t_now_list, th_end, x, m): #选取两台运行时间最长的停下来
    t_now = t_now_list[-1]
    t_now_list.append(th_end)
    dt = th_end - t_now
    t_record_now = [t_record.iloc[i,-1] + dt for i in range(6)]
    m = 2*(m%2)
    index_stop = [m, m+1]
    t_record[x+1] = t_record_now
    s_record_now = [1 for i in range(6)]
    for i in index_stop:
        s_record_now[i] = 3
    s_record[x+1] = s_record_now
    x+=1
    return t_record, s_record, t_now_list, x

def cost_low_pm(number, cost_rand):
    cost_sum = 0
    cost_rand = cost_rand[:number]
    large = sum(cost_rand>0.8)
    cost_sum += large*13
    cost_sum += (number-large)*8
    return cost_sum

def cost_high_fault(number, cost_rand):
    cost_sum = 0
    cost_day = 200 #宕机损失，一台泵一天
    cost_rand = cost_rand[:number]
    large = sum(cost_rand>0.8)
    cost_sum += large*(30+cost_day*30)
    cost_sum += (number-large)*(10+cost_day*15)        
    return cost_sum

def cost_low_fault(number,cost_rand):
    cost_sum = 0
    cost_rand = cost_rand[:number]
    large = sum(cost_rand>0.8)
    cost_sum += large*30
    cost_sum += (number-large)*10
    return cost_sum


def cost_downtime(hours):
    cost_day = 200
    day = hours/24
    cost = day*cost_day
    return cost
        

#抽样1，表示前面四台泵的寿命
lmda1 = 1/8000
k_wbl1 = 5
rand1 = np.random.rand(300000)
life1 = []
for i in range(len(rand1)):
    life1.append(sampling(i,rand1, lmda1, k_wbl1))
    
#抽样2，表示后面两台泵的寿命
lmda2 = 1/10000
life2 = []
for i in range(len(rand1)):
    life2.append(sampling(i, rand1, lmda2, k_wbl1))
    
year = 25 #控制计算年限 
initial_time = 5880
th_start_list = [initial_time+i*8760 for i in range(year)]
th_end_list = [initial_time+2880+i*8760 for i in range(year)]
t_table = pd.DataFrame()
t_table["th_start"] = th_start_list
t_table["th_end"] = th_end_list

tp_list = np.linspace(6000,14000,11)

cost_list = []
t_low_overlap_list = []
tf_low_list = []
tf_high_list = []
tp_count_list = []
Availability = [] #记录可用度
Availability_high = []
#maintenance_t = 720 #一次维修花费平均时间
cycle_times = 1
a = time.time()

for tp_time in tp_list:
    tp = tp_time
    y1 = 0
    y2 = 0
    y4 = 0
    th_time = 2880
    tf_low = 0
    tf_high = 0
    tp_count = 0
    t_low_overlap = []
    t_high_overlap =[]
    
    for times in range(cycle_times): #多次循环求平均数 
        t_record = pd.DataFrame([0 for i in range(6)])
        s_record = pd.DataFrame([1, 1, 1, 1, 3, 3])
        t_fault_list = [life1[i+y1] for i in range(4)]  #最开始的列表里面都有预期故障记录了
        y1 += 4
    
        for i in range(2):
            t_fault_list.append(life2[i+y2])
            y2 += 1
    
        x = 0
        t_now_list = [0]
        operation = 2
        for m in range(len(t_table)):
            th_start = t_table.iloc[m,0]
            
            index_using = [i for i in range(6) if s_record.iloc[i, -1] == 1]
            t_record_using = t_record.iloc[index_using,-1]
            t_fault_using = np.array(t_fault_list)[index_using]
            dtf = (t_fault_using-t_record_using).tolist()
            dtp = (tp-t_record_using).tolist()
            dtf_min = min(dtf)
            index_fault = index_using[dtf.index(min(dtf))]
            dtp_min =min(dtp)
            index_pm = index_using[dtp.index(min(dtp))]
            t_next_fault = t_now_list[-1] + dtf_min              
            t_next_tp = t_now_list[-1] + dtp_min
            if t_next_tp < t_next_fault:
                operation = 1
                t_next_fault = t_next_tp
            else:
                operation = 2
    
            times_stop_count_temp = 0 #用来统计在一次低负荷运行期间发生了几次故障
            t_low_stop_temp =[]
            th_start_temp = th_start - 1440
            while t_next_fault < th_start_temp:
                times_stop_count_temp += 1
                t_low_stop_temp.append(t_next_fault)
                if operation == 2:
                    tf_low += 1
                    t_record, s_record, t_fault_list, t_now_list, x, y1, y2 = fault_in_low(t_record, s_record, t_fault_list, t_now_list, th_start, x, y1, y2)
                else:
                    tp_count += 1
                    t_record, s_record, t_fault_list, t_now_list, x, y1, y2 = pm_in_low(t_record, s_record, t_fault_list, t_now_list,th_start, x, y1, y2)
                index_using = [i for i in range(6) if s_record.iloc[i, -1] == 1]
                t_record_using = t_record.iloc[index_using,-1]
                t_fault_using = np.array(t_fault_list)[index_using]
                dtf = (t_fault_using-t_record_using).tolist()
                dtp = (tp-t_record_using).tolist()
                dtf_min = min(dtf)
                index_fault = index_using[dtf.index(min(dtf))]
                dtp_min =min(dtp)
                index_pm = index_using[dtp.index(min(dtp))]
                t_next_fault = t_now_list[-1] + dtf_min              
                t_next_tp = t_now_list[-1] + dtp_min
                if t_next_tp < t_next_fault:
                    operation = 1
                    t_next_fault = t_next_tp
                else:
                    operation = 2
            #判定重叠部分的费用
            if times_stop_count_temp >= 3:
                for index_temp in range(times_stop_count_temp-2):
                    rand_temp = rand1[y4]
                    y4 += 1
                    if rand_temp > 0.8:
                        maintenance_t = 30*24
                    else:
                        maintenance_t = 15*24
                    if t_low_stop_temp[index_temp+2] <= t_low_stop_temp[index_temp] + maintenance_t:
                        t_overlap = t_low_stop_temp[index_temp] + maintenance_t - t_low_stop_temp[index_temp+2]
                        t_low_overlap.append(t_overlap)
    
                        
            th_end = t_table.iloc[m,1]
            th_start_list = [th_start_temp+i*240 for i in range(6)] #假设只有维修四台泵的维修时间
            for t_now in th_start_list:
                index_using = [i for i in range(6) if s_record.iloc[i,-1] == 1]
                t_record_temp = t_record.iloc[index_using,-1].tolist()
                index_pm = index_using[t_record_temp.index(max(t_record_temp))]
                dt_plus = t_now - t_now_list[-1]#上次时间节点到现在的时间差值
                t_pm_now = t_record.iloc[index_pm,-1] + dt_plus #表示最大运行时间的那台泵的运行时间
                if t_pm_now+(th_end-t_now)+th_time >= tp:
                    tp_count += 1
                    s_record_temp = s_record.iloc[:,-1]
                    index_spare = [i for i in range(6) if s_record.iloc[i, -1] != 1]
                    t_operation_spare = t_record.iloc[index_spare,-1].tolist()
                    index_change = index_spare[t_operation_spare.index(min(t_operation_spare))]  #把备件中，运行时间较大的继续停机
                    if index_pm == 5 or index_pm == 4:
                        t_fault_list[index_pm] = life2[y2]
                        y2 += 1
                    else:
                        t_fault_list[index_pm] = life1[y1]
                        y1 += 1
                    s_record_temp[index_change] = 1
                    s_record_temp[index_pm] = 4 #4表示维修
                    s_record[x+1] = s_record_temp
                    
                    t_record_temp = t_record[x].tolist() #不改成list会改变原来的值
                    for i in index_using:
                        t_record_temp[i] += dt_plus       
                    t_record_temp[index_pm] = 0
                    t_record[x+1] = t_record_temp
                    x += 1
                    t_now_list.append(t_now)                 
            #还得考虑在这一段时间内是不是有故障发生，做简化处理                           
                t_record_temp = t_record.iloc[:,-1]
                index_tf = [i for i in range(6) if t_fault_list[i]-t_record_temp[i] <= 0]
                if len(index_tf) > 0:
                    tf_low += len(index_tf)
                    for index_tflow in index_tf:
                        t_record.iloc[index_tflow,-1]=0
                        if index_tflow == 5 or index_tflow == 4:
                            t_fault_list[index_tflow] = life2[y2]
                            y2 += 1
                        else:
                            t_fault_list[index_tflow] = life1[y1]
                            y1+=1
            
            t_record, s_record, t_now_list, x = low_to_high(t_record, s_record, t_now_list, th_start, x)           
            dtf_high = [t_fault_list[i] - t_record.iloc[i, -1] for i in range(6)]
            dtf_high_min = min(dtf_high)
            t_next_fault = t_now_list[-1] + dtf_high_min
            tf_high_temp = 0
            while t_next_fault < th_end:
                tf_high_temp += 1
                tf_high += 1
                t_record, s_record, t_fault_list, t_now_list,t_next_fault, x, y1, y2 = fault_in_high(t_record, s_record, t_fault_list, t_now_list,th_end, x, y1, y2)
            t_high_overlap.append(tf_high_temp*360)
            t_record, s_record, t_now_list, x = high_to_low(t_record, s_record, t_now_list, th_end, x,m)    

    tf_high_list.append(tf_high/cycle_times)
    tf_low_list.append(tf_low/cycle_times)
    tp_count_list.append(tp_count/cycle_times)
    t_stop = sum(t_low_overlap)/cycle_times/4
    t_low_overlap_list.append(t_stop*4+0.00001)
    t_high_stop = sum(t_high_overlap)/cycle_times/6
    Availability.append((5880*25-t_stop)/5880/25)
    Availability_high.append((2880*25-t_high_stop)/2880/25)

  

b = time.time()
print("用时",b-a)
cost_rand = rand1
for i in range(len(tp_list)):
    tf_high_times = int(tf_high_list[i]*cycle_times)
    tf_low_times = int(tf_low_list[i]*cycle_times)
    tp_times = int(tp_count_list[i]*cycle_times)
    hrs = t_low_overlap_list[i]*cycle_times
    e1 =  cost_high_fault(tf_high_times, cost_rand)
    e = e1+ cost_low_fault(tf_low_times,cost_rand) +cost_low_pm(tp_times,cost_rand) + cost_downtime(hrs)
    cost_list.append(e/cycle_times)
df = pd.DataFrame([tf_high_list,tf_low_list,tp_count_list,Availability,Availability_high,tp_list,cost_list],
                  index=["tf_high","tf_low","tp_count","低负荷可用度", "高负荷期可用度","预防性维修限值","费用"])   

#df.to_csv("6000, 14000, 161,500times,k=5,24491s=6.8h,17-done")
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(tp_list,tf_high_list,label="高负荷期故障次数") 
plt.plot(tp_list,tf_low_list,label="低负荷期故障次数") 
plt.plot(tp_list,tp_count_list,label="预防性维修次数")
plt.xlabel("预防性维修限值")
plt.ylabel("平均次数")
plt.legend()
plt.show() 

plt.figure()
plt.plot(tp_list,cost_list)
#plt.ylim(0,250000)
plt.xlabel("预防性维修限值")
plt.ylabel("费用")
plt.show()

plt.figure()
plt.plot(tp_list,Availability,label="低负荷期可用度")
plt.plot(tp_list,Availability_high,label="高负荷期可用度")
plt.legend()
plt.ylim(0.9,1.1)
plt.xlabel("预防性维修限值")
plt.ylabel("可用度")
plt.show()



#就使用多项式拟合会比较好好看
x = tp_list
y = cost_list
z1 = np.polyfit(x, y, 5) # 用3次多项式拟合
p1 = np.poly1d(z1)
yvals=p1(x) # 也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()

