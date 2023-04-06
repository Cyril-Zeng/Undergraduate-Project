#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

from docplex.mp.model import Model
import  random
import numpy as np
from IPython.display import display
import pandas as pd
# 参数
M = 50000
n = 20 #顾客数目
n1 = round(n*0.5) #一类顾客数目
n2 = round(n*0.3) #二类顾客数目
n3 = n-n1-n2 #三类顾客数目
v_num = 4 #车辆数目
c1 = 30 #单位固定成本
c2 = 1 #单位变动成本
c3 = 5 #工作量均衡度的惩罚因子
p = 1 # 货物单价
eta = 0.5 # 损耗系数
K = [i for i in range(1,v_num+1)] #车辆集
N = [i for i in range(1, n+1)] #顾客集
V = [i for i in range(0,n+2)] #点集，包括集散中心和顾客点
A = [(i,j) for i in V for j in V ] #弧集
D = np.trunc(np.random.uniform(0,20,n)) #顾客需求集 单位：kg
D = np.insert(D,0,0)
D = np.insert(D,n+1,0)
Q = 50
st1 = np.random.uniform(1,2,n1)#服务时长
st1 = np.around(st1,decimals=1)
st2 = np.random.uniform(2,5,n2)#服务时长
st2 = np.around(st2,decimals=1)
st3 = np.random.uniform(5,10,n3)#服务时长
st3 = np.around(st3,decimals=1)
st = np.hstack((st1,st2,st3))
st = np.insert(st,0,0)
st = np.insert(st,n+1,0)


#时间窗
L = 480
random.seed(2)
tc = np.random.uniform(60,200,n) # 随机生成服从均匀分布
tc = np.trunc(tc)  #取整，保留整数部分
tl = np.random.uniform(60,120,n) # 随机生成服从均匀分布
tl = np.trunc(tl)  #取整，保留整数部分
a = [tc[i]-0.5*tl[i] for i in range(n)]
a.insert(0,0)
a.insert(n+1,0)
b = [tc[i]+0.5*tl[i] for i in range(n)]
b.insert(0,0)
b.insert(n+1,L)
time_window = [] 
for i in range(len(a)):
    time_window.append([a[i],b[i]])
    
#位置情况
random.seed(5)
loc = [[0,0]]+ [[random.randint(-20, 20) for j in range(1, 3)] for i in range(1,n+1)]+[[0,0]]
loc = np.array(loc)
# print(loc)
#画坐标散点图
import matplotlib.pyplot as plt
x_loc = loc[:,0]
y_loc = loc[:,1]
plt.scatter(x_loc,y_loc,c='b')
plt.plot(x_loc[0], y_loc[0],c='r',marker='s')
for i in N:
    plt.annotate('%d'%i,(x_loc[i]+0.5,y_loc[i]))
plt.annotate('%d'%0,(x_loc[0]+0.5,y_loc[0]))
import pylab
pylab.show()

#各节点间的距离
d = {(i,j):abs(loc[i][0]-loc[j][0])+abs(loc[i][1]-loc[j][1]) for i,j in A}
# print(d)

# 时间窗数据预处理
data = {'客户i':V,'X坐标':x_loc,'Y坐标':y_loc,'需求量':D,'服务时间窗':time_window}
data = pd.DataFrame(data)
display(data)
data_df = data.sort_values(by='服务时间窗',ascending=1)
# print(data_df)
data_2 = data_df.loc[n+1]
# print(data_2)
data_df = data_df.drop(labels=n+1)
data_df = data_df.append(data_2)
display(data_df)

tw_1 = list(data_df['服务时间窗'])
tw_2 = []
order = list(data_df['客户i'])
# print(tw_1,order)
n_cycle=1
# print(tw_1)
while tw_1 != tw_2   :
    if n_cycle == 1 :
        tw_2 = tw_1
    else:
        tw_1 = tw_2
#     print('tw_1:{0}'.format(tw_1))
#     print('tw_2:{0}'.format(tw_2))
    for i in range(1,n+1):
        E1 = 200000
        E1 = 200000
        L1 = 0
        L1 = 0
        i1 = order[i]
        for j in range(i):
            j1 = order[j]
            e1 = tw_2[j][0] + d[j1,i1]
            E1 = min(E1,e1)
            l1 = tw_2[j][1] + d[j1,i1]
            L1 = max(L1,l1)
#             print('E1：{0}，{1}'.format(j,E1))
#             print('tw_2:{0}'.format(tw_2))
        for j in range(i+1,n+1):
            j1 = order[j]
            e2 = tw_2[j][0] - d[i1,j1]
            E2 = min(E1,e2)
            l2 = tw_2[j][1] - d[i1,j1]
            L2 = max(L1,l2)
            
#             print('i:{0},j：{1},e2:{2}'.format(i,j,e2))
#             print('tw_2:{0}'.format(tw_2))
        tw_2[i][0] = max(tw_2[i][0],min(tw_2[i][1],E1))
        tw_2[i][0] = max(tw_2[i][0],min(tw_2[i][1],E2))
        tw_2[i][1] = min(tw_2[i][1],max(tw_2[i][0],L1))
        tw_2[i][1] = min(tw_2[i][1],max(tw_2[i][0],L2)) 
#         print('tw2:{0}'.format(tw_2))
    n_cycle += 1  
print('n_cycle:{0}'.format(n_cycle))
data_df['服务时间窗'] = tw_2
data_df = data.sort_values(by='客户i',ascending=1)
display(data_df)
for i in range(n+1):
    a[i] = data_df.at[i,'服务时间窗'][0]
    b[i] = data_df.at[i,'服务时间窗'][1]
# print(a,b)

# 求工作时长极差数据预处理
from itertools import product
cc = product(K,repeat=2)
cc = [x for x in cc if x[0]!=x[1]]
# print(cc)


# 模型
mdl = Model(name='basic model')
#决策变量
x_kij = [(k,i,j) for k in K for i,j in A]
x = mdl.binary_var_dict(x_kij, name='x')
#print(x)
t_ki = [(k,i) for k in K for i in V]
t = mdl.continuous_var_dict(t_ki,lb=0,name='t')
y = mdl.continuous_var(lb=0)

#目标函数
mdl.minimize(c1* mdl.sum(x[k,0,j] for k in K for j in N) +             c2* mdl.sum(d[i,j]* x[k,i,j] for k in K for i,j in A) +              p*eta*mdl.sum(D[i]*t[k,i] for k in K for i in N) +             c3*y
            )


#约束条件
mdl.add_constraints(mdl.sum(D[i]*x[k,i,j] for i in N for j in V) <= Q for k in K)
mdl.add_constraints(mdl.sum(x[k,i,j] for k in K for j in V) == 1 for i in N)
mdl.add_constraints((mdl.sum(x[k,i,j] for i in V)-mdl.sum(x[k,j,i] for i in V)) == 0 for k in K for j in N)
mdl.add_constraints(mdl.sum(x[k,0,j] for j in V) == 1 for k in K)
mdl.add_constraints(mdl.sum(x[k,i,n+1] for i in V) == 1 for k in K)
mdl.add_constraints(x[k,i,i] == 0 for i in V for k in K)
mdl.add_constraints((t[k,i]+st[i]+d[i,j]) <= t[k,j]+M*(1-x[k,i,j]) for i in V for j in V for k in K)
mdl.add_constraint(mdl.sum(x[k,0,j] for j in N for k in K) <= v_num)
mdl.add_constraints(t[k,0] == 0 for k in K)
mdl.add_constraints(a[i]*mdl.sum(x[k,i,j] for j in V) <= t[k,i] for k in K for i in N)
mdl.add_constraints(b[i]*mdl.sum(x[k,i,j] for j in V) >= t[k,i] for k in K for i in N)
mdl.add_constraints(t[k,i] >= 0 for k in K for i in [n+1])
mdl.add_constraints(t[k,i] <= L for k in K for i in [n+1])
mdl.add_constraints(y >= t[z[0],n+1]-t[z[1],n+1] for z in cc)

#求解
solution = mdl.solve()
print(solution)
print(solution.solve_details)

# 结果可视化
import matplotlib.pyplot as plt
x_loc = loc[:,0]
y_loc = loc[:,1]
plt.scatter(x_loc,y_loc,c='b')
plt.plot(x_loc[0], y_loc[0],c='r',marker='s')
for i in N:
    plt.annotate('%d'%i,(x_loc[i]+0.5,y_loc[i]))
plt.annotate('%d'%0,(x_loc[0]+0.5,y_loc[0]))

c = [0,'k','g','b','r','y','c','pink','grey','m','brown']
active_arcs=[ a for a in x_kij if x[a].solution_value ]
# print(active_arcs)
for k,i,j in active_arcs:
    plt.plot((x_loc[i], x_loc[j]), (y_loc[i], y_loc[j]), c=c[k])
import pylab
pylab.show()

route=[]
for k in K:
    route.append({})
    for i in V:
        if t[k,i].solution_value:
            route[-1][i] = t[k,i].solution_value
    route[-1]=sorted(route[-1].items(), key=lambda d:d[1], reverse = False)
# print(route)
for k in K:
    print('车辆{0}'.format(k))
    a=[0]
    for i in range(len(route[k-1])):
        a.append(route[k-1][i][0])
    print(a)
        
                   






# In[ ]:




