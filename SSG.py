# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:29:19 2023

@author: USER
"""
#pip install random
#pip install dtaidistance
#pip install numpy

from random import choice
from random import shuffle
from dtaidistance import dtw
import numpy as np


def frechet(x,timeseries):
    dtwdis = []
    l = len(timeseries)
    for i  in range(l):
        distance = dtw.distance(x.reshape(len(x),1),np.array([item for sublist in timeseries[i] for item in sublist]).reshape(len(np.array([item for sublist in timeseries[i] for item in sublist])),1))
        dtwdis.append(distance)
    F = (1/l)*(sum([x**2 for x in dtwdis]))
    return F

##############################################################################################################
def SSG_avg(z,timeseries):
    l = range(0,len(timeseries))
    z = z.reshape(len(z),1)
    zbest = z#設定起始最佳平均
    t = 1#迭代次數
    ni = 0.05#數學習率
    while frechet(z,timeseries)!= min(frechet(z, timeseries),frechet(zbest, timeseries)):
        zbest = z
        l2 = list(l)
        shuffle(l2)
        for k in l2:
            ts = np.array([item for sublist in timeseries[k] for item in sublist]).reshape(len(np.array([item for sublist in timeseries[k] for item in sublist])))
            warpingpath = dtw.warping_path(z,ts)
            w = np.zeros((len(z),len(ts)))
            for i in range(len(warpingpath)):
                w[warpingpath[i]] = 1
            v = np.zeros((len(z),len(ts)))
            for i in range(len(z)):
                v[i,i] = sum(w[i])
            z1 = z-(2*ni*(v.dot(z)-w.dot(ts)))
            if frechet(z1, timeseries) == min(frechet(z1, timeseries),frechet(z, timeseries)): 
                z = z1
        t = t+1
        if t <= len(timeseries):
            ni = (ni)-((0.05-0.005)/len(timeseries))
        elif t > len(timeseries):
            ni = 0.005#最終學習率
    else:
        return zbest
#############################################################################################################

def MSE(x):
    CX = []
    C = {}
    D = np.zeros((len(x)+1,len(x)+1))
    C['(1,1)'] = x[0]
    D[1,1] = 0
    for i in range(1,len(x)+1):
        for m in range(1,i+1):
            jstar = 0
            dstar = float("inf")
            ustar = []
            for j in range(m,i+1):
                #計算平均值
                u1=[]
                for p in range(j,i+1):
                    u1.append(x[p-1])
                u = sum(u1)/(i-j+1)
                #計算距離
                d1 = []
                for q in range(j,i+1):
                    d1.append((x[q-1]-u)*(x[q-1]-u))
                d = sum(d1)
                if D[j-1,m-1]+d < dstar:
                    dstar = D[j-1,m-1]+d
                    ustar.append(u)
                    jstar = j
            D[i,m] = dstar
            if i > 1 & m > 1 :
                C['({},{})'.format(i,m)] = C['({},{})'.format(i-1,m-1)]
                for number in ustar:
                    C['({},{})'.format(i,m)].append(number)
            else:
                C['({},{})'.format(i,m)] = []
                for number in ustar:
                    C['({},{})'.format(i,m)].append(number)
    for index in range(1,len(x)+1):
        CX.append(list(C['({},{})'.format(len(x),index)]))
    return CX

####################################################################################

def AC(timeseries):
    randonnumber = range(0,len(timeseries))
    #隨機抽取起始值
    z = np.array([item for sublist in timeseries[choice(randonnumber)] for item in sublist]).reshape(len(np.array([item for sublist in timeseries[choice(randonnumber)] for item in sublist])))
    zbest = z#最佳平均
    lstar = len(zbest)#最佳平均長度
    fstar = frechet(zbest, timeseries)
    while True:
        z = SSG_avg(z,timeseries)
        c = MSE(z)
        f = []
        for i in range(len(c)):
            f.append(frechet(np.array(c[i]), timeseries))
        z = np.array([item for sublist in c[f.index(min(f))] for item in sublist])
        f = min(f)
        if f<fstar or(f==fstar and len(z)<lstar):
            fstar = f
            lstar = len(z)
            zbest = z
        else:
            return zbest
            break


##########################################################################################
def aligns(data,mean):
    aline = []
    for array_a in data:
        array_a = array_a.T
        dtwpath = dtw.warping_path(mean,array_a)
        meanlist = [iterm[0] for iterm in dtwpath]
        goallist = [iterm[1] for iterm in dtwpath]
        line = []
        for i in range(array_a.shape[1]):
            if meanlist.count(i) > 1:#判斷mean是否重複
                c = i
                h = [k for k,x in enumerate(meanlist) if x == c]#找出mean重複的位置
                p = [goallist[k] for k in h]#找出rohs對應的位置
                n = np.array([array_a[:,k] for k in p])
                point = np.mean(n,axis=1)
                line.append(point)
            else:
                N = array_a[:,i]
                line.append(N)
        aline.append(line)
    aline=np.array(aline)
    return aline
###############################################################################################
def aligns_1d(a,mean):
    align_list = []#對其後的ROHS
    for j in range(len(a)):
        serise = a[j].T
        dtwpath = dtw.warping_path(mean,serise)
        meanlist = [iterm[0] for iterm in dtwpath]
        xlist = [iterm[1] for iterm in dtwpath]
        line = []
        for i in range(len(serise)):
            if meanlist.count(i) > 1:#判斷mean是否重複
                c = i
                h = [k for k,x in enumerate(meanlist) if x == c]#找出mean重複的位置
                p = [xlist[k] for k in h]#找出rohs對應的位置
                n = [serise[k] for k in p]
                point = sum(n)/len(n)
                line.append(point)
            else:
                N = serise[i]
                line.append(N)
                
        align_list.append(line)
    return np.array(align_list)

    
