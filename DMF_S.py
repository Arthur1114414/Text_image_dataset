# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:21:38 2024

@author: USER
"""
#pip install cv2
#pip install os
#pip install numpy
#pip install math
#pip install scipy
#pip install sklearn

import cv2
import os
import numpy as np
import math
from scipy.signal import find_peaks
from SSG import AC,aligns_1d
from imgRotate import rotate_img
import itertools
""""""""""    DMF-S    """""""""
class DMF_S():

    ### P_dir is the directory of Positive dataset
    ### N_dir is the directory of Negative dataset
    ## ang = rotation angle
    def __init__(self,P_dir,N_dir,ang=0):
        """ Step 1 """
        """ Read the dataset """
        ### path = the directory of dataset
        def read(path):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            datat = []
            for path in PATH:
                img = cv2.imread(path)
                resize_img = cv2.resize(img,(200,200))#resize
                resize_img = rotate_img(resize_img, ang)#rotate
                gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)#gray
                blur = cv2.medianBlur(gray,5)#blur
                canny_img = cv2.Canny(blur, 50, 100)#canny
                serise = np.sum(canny_img,axis=0)
                serise_t = np.sum(canny_img.T,axis=0)
                data.append([serise])
                datat.append([serise_t])
            data=np.array(data)
            datat=np.array(datat)
            return data,datat
        
        
        a,b = read(P_dir)## Positive dataset: a is x-axis projectin & b is y-axis projectin
        o,g = read(N_dir)## Negative dataset: o is x-axis projectin & g is y-axis projectin
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 2 """
        mean_x=AC(a).reshape(-1,1)### Computes the mean of x-axis projectin(positive)
        mean_y=AC(b).reshape(-1,1)### Computes the mean of y-axis projectin(positive)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 3 """
        """ Alignment """
        align_px=aligns_1d(a,mean_x)### positive aligns to mean of x-axis
        align_py=aligns_1d(b,mean_y)### positive aligns to mean of y-axis
        align_nx=aligns_1d(o,mean_x)### negative aligns to mean of x-axis
        align_ny=aligns_1d(g,mean_y)### negative aligns to mean of y-axis
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 4 """
        """ Create the envelope """
        cov = []
        covy = []
        for i in range(len(mean_x)):
            cov.append([math.sqrt(np.var(align_px[:,i,0]))])
            covy.append([math.sqrt(np.var(align_py[:,i,0]))])
        
        ## align_data = The dataset of alignment serise 
        ## bound = The upper bound or lower bound
        ## judge = Determine the type of boundary ("upper" or "lower")
        #### data_index = The index of the unfilte negative data in previous step (for smelt)
        #### range_i = The index of peaks from serise (for smelt)
        def compress(align_data,bound,judge,data_index=False,range_i=False,count=False):
            nzn = []#compressed serise
            nzr =[]#non-zero ratio
            if data_index == False:
                data_index=range(len(align_data))
            if range_i == False:
                counter=[]
                for i in data_index:
                    x = align_data[i]
                    zero = []
                    if judge == "upper":
                        for j in range(len(x)):
                            if x[j] > bound[j]:
                                zero.append(1)
                            elif x[j] <= bound[j]:
                                zero.append(0)
                    elif judge == "lower":
                        for j in range(len(x)):
                            if x[j] < bound[j]:
                                zero.append(1)
                            elif x[j] >= bound[j]:
                                zero.append(0)
                    nzn.append(zero)
                    nzr.append(np.mean(zero))
                    counter.append(i)
            else:
                counter=[]
                for i in data_index:
                    x = align_data[i]
                    zero = []
                    if judge == "upper":
                        for j in range_i:
                            if x[j] > bound[j]:
                                zero.append(1)
                            elif x[j] <= bound[j]:
                                zero.append(0)
                    elif judge == "lower":
                        for j in range_i:
                            if x[j] < bound[j]:
                                zero.append(1)
                            elif x[j] >= bound[j]:
                                zero.append(0)
                    nzn.append(zero)
                    nzr.append(np.mean(zero))
                    counter.append(i)
            if count == False:
                return nzn,nzr
            elif count == True:
                return nzn,nzr,counter
                
        """Create the upper bound and lower bound"""
        def boundary(p,n,mean,cov,judge):
            B=0
            Y=0
            K=0
            pnzr=0
            nnzr=0
            pnzn=0
            nnzn=0
            for k in np.round(np.arange(0.1,7.1,0.1),1):
                bound = []
                for i in range(len(mean)):
                     h = np.array(mean[i])+(np.array(cov[i]) * k)
                     bound.append(h)
                PN,P = compress(p,bound,judge)
                NN,N = compress(n,bound,judge)
                P,N = pred(P, N, max(P))
                _,yd,_,_=pr(P, N)
                if yd > Y:
                    Y=yd
                    B=bound
                    pnzr=P
                    nnzr=N
                    pnzn=PN
                    nnzn=NN
                    K=k
            return K,B,pnzn,nnzn,pnzr,nnzr
                 
        ### The upper bound & lower bound of projection x
        k1,ubx,nzn_u_px,nzn_u_nx,nzr_u_px,nzr_u_nx = boundary(align_px, align_nx, mean_x, cov, judge="upper")
        k11,lbx,nzn_l_px,nzn_l_nx,nzr_l_px,nzr_l_nx = boundary(align_px, align_nx, mean_x, cov, judge="lower")
        k2,uby,nzn_u_py,nzn_u_ny,nzr_u_py,nzr_u_ny = boundary(align_py, align_ny, mean_y, covy, judge="upper")
        k22,lby,nzn_l_py,nzn_l_ny,nzr_l_py,nzr_l_ny = boundary(align_py, align_ny, mean_y, covy, judge="lower")
        """"""""""""""""""""""""""""""""""""""
        
        def out_of(lst, a, judge="max"):
            indexes = []
            if judge == "max":
                for i, value in enumerate(lst):
                    if value > max(a):
                        indexes.append(i)
                    else:
                        continue
            elif judge =="min":
                for i, value in enumerate(lst):
                    if value < min(a):
                        indexes.append(i)
                    else:
                        continue
            return indexes
        
        delindex = list(set(out_of(nzr_l_nx, nzr_l_px)+out_of(nzr_u_nx, nzr_u_px)+out_of(nzr_l_ny, nzr_l_py)+out_of(nzr_u_ny, nzr_u_py)))
        notdel = [index for index, element in enumerate(align_nx) if index not in delindex]
        
        """ Step 6 """
        """ smelt """
        # To make sure "TAU" is functioning properly
        def tran0(a):
            a[np.isnan(a)] = 0
            a[np.isinf(a)] = 0
            return a
        
        # To find the value of the unfilte negative data in previous step
        def undelet_value(goal,compare):
            return [value for index,value in enumerate(goal) if value not in compare]
        
        # To find the index of the unfilte negative data in previous step
        def inside(goal,compare):
            return [value for index,value in enumerate(goal) if index in compare]
        
        # The "smelt" step of DMF-S
        def smelt(Pnzn,Nnzn,notdelet,xy,ul,qrange=np.round(np.arange(0,5.1,0.01),1)):
            if xy == "x": 
                data_p = align_px
                data_n = align_nx
                judge = ul
                if ul == "upper":
                    bound = ubx
                elif ul == "lower":
                    bound = lbx
            elif xy == "y":
                data_p = align_py
                data_n = align_ny
                judge = ul
                if ul == "upper":
                    bound = uby
                elif ul == "lower":
                    bound = lby
            P = sum(np.array(Pnzn))/len(Pnzn)
            N = sum(np.array(Nnzn))/len(Nnzn)
            tau = tran0(N/P)
            result = {}
            for q in qrange:
                peaks,_ = find_peaks(tau,height=q)
                _,pnzr = compress(data_p, bound, judge, range_i=peaks)
                
                _,nnzr,ncounter = compress(data_n, bound, judge, data_index=notdelet, range_i=peaks, count=True)
                
                nzr = out_of(nnzr, pnzr)
                result["q:{}".format(q)] = inside(ncounter, nzr)
            
            q = float(max(result, key=lambda k: len(result[k]))[3:6])
            notq = undelet_value(notdel,result[max(result, key=lambda k: len(result[k]))])
            peaks,_ = find_peaks(tau,height=q)
            _,pnzr = compress(data_p, bound, judge, range_i=peaks)
            _,nnzr = compress(data_n, bound, judge, range_i=peaks)
            return pnzr,nnzr,notq,peaks
        
        snzr_u_px,snzr_u_nx,notdel,peaks_ux = smelt(nzn_u_px, nzn_u_nx, notdel, "x", "upper")
        snzr_l_px,snzr_l_nx,notdel,peaks_lx = smelt(nzn_l_px, nzn_l_nx, notdel, "x", "lower")
        snzr_u_py,snzr_u_ny,notdel,peaks_uy = smelt(nzn_u_py, nzn_u_ny, notdel, "y", "upper")
        snzr_l_py,snzr_l_ny,notdel,peaks_ly = smelt(nzn_l_py, nzn_l_ny, notdel, "y", "lower")
        
        """ Step 5 & Step 7 """
        """ Decision tree """
        
        def pred(P,N,threshold):
            a = []
            b = []
            for i in P:
                if i > threshold:
                    a.append(0)
                elif i <= threshold:
                    a.append(1)
            for j in N:
                if j > threshold:
                    b.append(0)
                elif j <= threshold:
                    b.append(1)
            a = np.array(a)
            b = np.array(b)
            return a,b
        
        def pr(a,b):#a是目標，b是非目標
            tp = sum(a)
            fn = sum([abs(k-1) for k in a])
            tn = sum([abs(l-1) for l in b])
            fp = sum(b)
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            accu = (tp+tn)/(tp+tn+fp+fn)
            youden = tpr - fpr
            return accu,youden,tpr,fpr
        
        def DTfilte(P,N):
            Y=0
            for i in sorted(set(P+[k for k in N if k <= max(P)])):
                ppred,npred = pred(P, N, i)
                _,yd,_,_ = pr(ppred, npred)
                if yd >= Y:
                    Y = yd
                    A = {i:[ppred,npred]}
                else:
                    pass
            t = list(A.keys())[0]
            a = np.array(A[t][0])
            b = np.array(A[t][1])
            return a,b,t
        
        
        A1,B1,t1 = DTfilte(nzr_u_px, nzr_u_nx)
        A2,B2,t2= DTfilte(nzr_l_px, nzr_l_nx)
        A3,B3,t11 = DTfilte(nzr_u_py, nzr_u_ny)
        A4,B4,t22 = DTfilte(nzr_l_py, nzr_l_ny)
        SA1,SB1,t3 = DTfilte(snzr_u_px, snzr_u_nx)
        SA2,SB2,t4 = DTfilte(snzr_l_px, snzr_l_nx)
        SA3,SB3,t33 = DTfilte(snzr_u_py, snzr_u_ny)
        SA4,SB4,t44 = DTfilte(snzr_l_py, snzr_l_ny)
        
        POSITIVE = np.where(np.mean([A1,A2,A3,A4,SA1,SA2,SA3,SA4],axis=0)<1,0,1)
        NEGATIVE = np.where(np.mean([B1,B2,B3,B4,SB1,SB2,SB3,SB4],axis=0)<1,0,1)
        reallabel_p = np.ones(len(POSITIVE))
        reallabel_n = np.zeros(len(NEGATIVE))
        classfy = {"real":np.concatenate((reallabel_p,reallabel_n)),"predict":np.concatenate((POSITIVE,NEGATIVE))}
        
        accuracy,youden,tpr,fpr=pr(POSITIVE, NEGATIVE)
        

        self.meanx = mean_x
        self.meany = mean_y
        self.k = {0:k1,1:k11,2:k2,3:k22}
        self.bound = {0:ubx,1:lbx,2:uby,3:lby}
        self.accuracy = accuracy
        self.youden = youden
        self.tpr = tpr
        self.fpr = fpr
        self.result = classfy
        self.nzr = {0:nzr_u_px,1:nzr_u_nx,2:nzr_l_px,3:nzr_l_nx,4:nzr_u_py,5:nzr_u_ny,6:nzr_l_py,7:nzr_l_ny}
        self.subnzr = {0:snzr_u_px,1:snzr_u_nx,2:snzr_l_px,3:snzr_l_nx,4:snzr_u_py,5:snzr_u_ny,6:snzr_l_py,7:snzr_l_ny}
        self.ang = ang
        self.peaks = {0:peaks_ux,1:peaks_lx,2:peaks_uy,3:peaks_ly}
        self.threshold = {0:t1,1:t2,2:t3,3:t4,4:t11,5:t22,6:t33,7:t44}
        
        
        
p="D:/ROHS"
n="D:/output/tran/Others"
test = DMF_S(p,n)



class DMF_A():

    ### P_dir is the directory of Positive dataset
    ### N_dir is the directory of Negative dataset
    ## ang = rotation angle
    def __init__(self,P_dir,N_dir,ang=0):
        """ Step 1 """
        """ Read the dataset """
        ### path = the directory of dataset
        def read(path):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            datat = []
            for path in PATH:
                img = cv2.imread(path)
                resize_img = cv2.resize(img,(200,200))#resize
                resize_img = rotate_img(resize_img, ang)#rotate
                gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)#gray
                blur = cv2.medianBlur(gray,5)#blur
                canny_img = cv2.Canny(blur, 50, 100)#canny
                serise = np.sum(canny_img,axis=0)
                serise_t = np.sum(canny_img.T,axis=0)
                data.append([serise])
                datat.append([serise_t])
            data=np.array(data)
            datat=np.array(datat)
            return data,datat
        
        
        a,b = read(P_dir)## Positive dataset: a is x-axis projectin & b is y-axis projectin
        o,g = read(N_dir)## Negative dataset: o is x-axis projectin & g is y-axis projectin
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 2 """
        mean_x=AC(a).reshape(-1,1)### Computes the mean of x-axis projectin(positive)
        mean_y=AC(b).reshape(-1,1)### Computes the mean of y-axis projectin(positive)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 3 """
        """ Alignment """
        align_px=aligns_1d(a,mean_x)### positive aligns to mean of x-axis
        align_py=aligns_1d(b,mean_y)### positive aligns to mean of y-axis
        align_nx=aligns_1d(o,mean_x)### negative aligns to mean of x-axis
        align_ny=aligns_1d(g,mean_y)### negative aligns to mean of y-axis
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 4 """
        """ Create the envelope """
        cov = []
        covy = []
        for i in range(len(mean_x)):
            cov.append([math.sqrt(np.var(align_px[:,i,0]))])
            covy.append([math.sqrt(np.var(align_py[:,i,0]))])
        
        ## align_data = The dataset of alignment serise 
        ## bound = The upper bound or lower bound
        ## judge = Determine the type of boundary ("upper" or "lower")
        #### data_index = The index of the unfilte negative data in previous step (for smelt)
        #### range_i = The index of peaks from serise (for smelt)
        def compress(align_data,bound,judge,data_index=False,range_i=False,count=False):
            nzn = []#compressed serise
            nzr =[]#non-zero ratio
            if data_index == False:
                data_index=range(len(align_data))
            if range_i == False:
                counter=[]
                for i in data_index:
                    x = align_data[i]
                    zero = []
                    if judge == "upper":
                        for j in range(len(x)):
                            if x[j] > bound[j]:
                                zero.append(1)
                            elif x[j] <= bound[j]:
                                zero.append(0)
                    elif judge == "lower":
                        for j in range(len(x)):
                            if x[j] < bound[j]:
                                zero.append(1)
                            elif x[j] >= bound[j]:
                                zero.append(0)
                    nzn.append(zero)
                    nzr.append(np.mean(zero))
                    counter.append(i)
            else:
                counter=[]
                for i in data_index:
                    x = align_data[i]
                    zero = []
                    if judge == "upper":
                        for j in range_i:
                            if x[j] > bound[j]:
                                zero.append(1)
                            elif x[j] <= bound[j]:
                                zero.append(0)
                    elif judge == "lower":
                        for j in range_i:
                            if x[j] < bound[j]:
                                zero.append(1)
                            elif x[j] >= bound[j]:
                                zero.append(0)
                    nzn.append(zero)
                    nzr.append(np.mean(zero))
                    counter.append(i)
            if count == False:
                return nzn,nzr
            elif count == True:
                return nzn,nzr,counter
                
        """Create the upper bound and lower bound"""
        def boundary(p,n,mean,cov,judge):
            B=0
            Y=0
            K=0
            pnzr=0
            nnzr=0
            pnzn=0
            nnzn=0
            for k in np.round(np.arange(0.1,7.1,0.1),1):
                bound = []
                for i in range(len(mean)):
                     h = np.array(mean[i])+(np.array(cov[i]) * k)
                     bound.append(h)
                PN,P = compress(p,bound,judge)
                NN,N = compress(n,bound,judge)
                P,N = pred(P, N, max(P))
                _,yd,_,_=pr(P, N)
                if yd > Y:
                    Y=yd
                    B=bound
                    pnzr=P
                    nnzr=N
                    pnzn=PN
                    nnzn=NN
                    K=k
            return K,B,pnzn,nnzn,pnzr,nnzr
                 
        ### The upper bound & lower bound of projection x
        k1,ubx,nzn_u_px,nzn_u_nx,nzr_u_px,nzr_u_nx = boundary(align_px, align_nx, mean_x, cov, judge="upper")
        k11,lbx,nzn_l_px,nzn_l_nx,nzr_l_px,nzr_l_nx = boundary(align_px, align_nx, mean_x, cov, judge="lower")
        k2,uby,nzn_u_py,nzn_u_ny,nzr_u_py,nzr_u_ny = boundary(align_py, align_ny, mean_y, covy, judge="upper")
        k22,lby,nzn_l_py,nzn_l_ny,nzr_l_py,nzr_l_ny = boundary(align_py, align_ny, mean_y, covy, judge="lower")
        """"""""""""""""""""""""""""""""""""""
        
        def out_of(lst, a, judge="max"):
            indexes = []
            if judge == "max":
                for i, value in enumerate(lst):
                    if value > max(a):
                        indexes.append(i)
                    else:
                        continue
            elif judge =="min":
                for i, value in enumerate(lst):
                    if value < min(a):
                        indexes.append(i)
                    else:
                        continue
            return indexes
        
        """ Step 5 """
        """ Decision tree """
        
        def pred(P,N,threshold):
            a = []
            b = []
            for i in P:
                if i > threshold:
                    a.append(0)
                elif i <= threshold:
                    a.append(1)
            for j in N:
                if j > threshold:
                    b.append(0)
                elif j <= threshold:
                    b.append(1)
            a = np.array(a)
            b = np.array(b)
            return a,b
        
        def pr(a,b):#a是目標，b是非目標
            tp = sum(a)
            fn = sum([abs(k-1) for k in a])
            tn = sum([abs(l-1) for l in b])
            fp = sum(b)
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            accu = (tp+tn)/(tp+tn+fp+fn)
            youden = tpr - fpr
            return accu,youden,tpr,fpr
        
        def DTfilte(P,N):
            Y=0
            for i in sorted(set(P+[k for k in N if k <= max(P)])):
                ppred,npred = pred(P, N, i)
                _,yd,_,_ = pr(ppred, npred)
                if yd >= Y:
                    Y = yd
                    A = {i:[ppred,npred]}
                else:
                    pass
            t = list(A.keys())[0]
            a = np.array(A[t][0])
            b = np.array(A[t][1])
            return a,b,t
        
        
        A1,B1,t1 = DTfilte(nzr_u_px, nzr_u_nx)
        A2,B2,t2= DTfilte(nzr_l_px, nzr_l_nx)
        A3,B3,t11 = DTfilte(nzr_u_py, nzr_u_ny)
        A4,B4,t22 = DTfilte(nzr_l_py, nzr_l_ny)

        
        POSITIVE = np.where(np.mean([A1,A2,A3,A4],axis=0)<1,0,1)
        NEGATIVE = np.where(np.mean([B1,B2,B3,B4],axis=0)<1,0,1)
        reallabel_p = np.ones(len(POSITIVE))
        reallabel_n = np.zeros(len(NEGATIVE))
        classfy = {"real":np.concatenate((reallabel_p,reallabel_n)),"predict":np.concatenate((POSITIVE,NEGATIVE))}
        
        accuracy,youden,tpr,fpr=pr(POSITIVE, NEGATIVE)
        

        self.meanx = mean_x
        self.meany = mean_y
        self.k = {0:k1,1:k11,2:k2,3:k22}
        self.bound = {0:ubx,1:lbx,2:uby,3:lby}
        self.accuracy = accuracy
        self.youden = youden
        self.tpr = tpr
        self.fpr = fpr
        self.result = classfy
        self.nzr = {0:nzr_u_px,1:nzr_u_nx,2:nzr_l_px,3:nzr_l_nx,4:nzr_u_py,5:nzr_u_ny,6:nzr_l_py,7:nzr_l_ny}
        self.ang = ang
        self.threshold = {0:t1,1:t2,2:t11,3:t22}



class DMF_S_test():
    def __init__(self,model,testp_dir,testn_dir,plot=False):
        ## path = path of the file
        ## ni = quantity of sampling 
        def read(path):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            datat = []
            for path in PATH:
                img = cv2.imread(path)
                resize_img = cv2.resize(img,(200,200))#resize
                resize_img = rotate_img(resize_img, angle=model.ang)#rotate
                gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)#gray
                blur = cv2.medianBlur(gray,5)#blur
                canny_img = cv2.Canny(blur, 50, 100)#canny
                serise = np.sum(canny_img,axis=0)
                serise_t = np.sum(canny_img.T,axis=0)
                data.append([serise])
                datat.append([serise_t])
            data=np.array(data)
            datat=np.array(datat)
            return data,datat
       
            
        align_px,align_py = read(testp_dir, model.ni)
        align_nx,align_ny = read(testn_dir, model.ni)
        def compressed(data):
            unz=[]
            lnz=[]
            for i in range(len(data)):
                x=data[i]
                upper = model.bound[0]
                lower = model.bound[1]
                upzs=0
                lowzs=0
                for j in range(len(x)):
                    if x[j] > upper[j]:
                        upzs+=1
                    elif x[j] < lower[j]:
                        lowzs+=1
                unz.append(upzs/len(x))
                lnz.append(lowzs/len(x))
            return unz,lnz
        def compressed_s(data,peaks):
            unz=[]
            lnz=[]
            for i in range(len(data)):
                x=data[i]
                upper = model.bound[0]
                lower = model.bound[1]
                upzs=0
                lowzs=0
                for j in peaks[0]:
                    if x[j] > upper[j]:
                        upzs+=1
                for j in peaks[1]:
                    if x[j] < lower[j]:
                        lowzs+=1
                unz.append(upzs/len(peaks[0]))
                lnz.append(lowzs/len(peaks[1]))
            return unz,lnz
        def pred(P,N,threshold):
            a = []
            b = []
            for i in P:
                if i > threshold:
                    a.append(0)
                elif i <= threshold:
                    a.append(1)
            for j in N:
                if j > threshold:
                    b.append(0)
                elif j <= threshold:
                    b.append(1)
            a = np.array(a)
            b = np.array(b)
            return a,b
        
        def pr(a,b):#a是目標，b是非目標
            tp = sum(a)
            fn = sum([abs(k-1) for k in a])
            tn = sum([abs(l-1) for l in b])
            fp = sum(b)
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            accu = (tp+tn)/(tp+tn+fp+fn)
            youden = tpr - fpr
            return accu,youden,tpr,fpr
        
        unz_px,lnz_px = compressed(align_px)
        unz_nx,lnz_nx = compressed(align_nx)
        unz_py,lnz_py = compressed(align_py)
        unz_ny,lnz_ny = compressed(align_ny)
        unzs_px,lnzs_px = compressed_s(align_px,[model.peaks[0],model.peaks[1]])
        unzs_nx,lnzs_nx = compressed_s(align_nx,[model.peaks[0],model.peaks[1]])
        unzs_py,lnzs_py = compressed_s(align_py,[model.peaks[2],model.peaks[3]])
        unzs_ny,lnzs_ny = compressed_s(align_ny,[model.peaks[2],model.peaks[3]])
        
        crupx,crunx = pred(unz_px,unz_nx,model.threshold[0])
        crlpx,crlnx = pred(lnz_px,lnz_nx,model.threshold[1])
        crupy,cruny = pred(unz_py,unz_ny,model.threshold[4])
        crlpy,crlny = pred(lnz_py,lnz_ny,model.threshold[5])
        crupsx,crunsx = pred(unzs_px,unzs_nx,model.threshold[2])
        crlpsx,crlnsx = pred(lnzs_px,lnzs_nx,model.threshold[3])
        crupsy,crunsy = pred(unzs_py,unzs_ny,model.threshold[6])
        crlpsy,crlnsy = pred(lnzs_py,lnzs_ny,model.threshold[7])
        
        POSITIVE = np.where(np.mean([crupx,crlpx,crupsx,crlpsx,crupy,crlpy,crupsy,crlpsy],axis=0)<1,0,1)
        NEGATIVE = np.where(np.mean([crunx,crlnx,crunsx,crlnsx,cruny,crlny,crunsy,crlnsy],axis=0)<1,0,1)
        reallabel_p = np.ones(len(POSITIVE))
        reallabel_n = np.zeros(len(NEGATIVE))
        classfy = {"real":np.concatenate((reallabel_p,reallabel_n)),"predict":np.concatenate((POSITIVE,NEGATIVE))}
        
        accuracy,youden,tpr,fpr=pr(POSITIVE, NEGATIVE)
        
        self.result = classfy
        self.accuracy = accuracy
        self.youden = youden
        self.tpr = tpr
        self.fpr = fpr
        
        if plot == True:
            def ulimit(array,how):
                for i in range(len(array)):
                    if sum(array[i])==how:
                        ii=i+1
                        return ii
            def pred_roc(nzr_p,nzr_n):
                AI=[]
                BI=[]
                for th in sorted(set(nzr_p+nzr_n)):
                    A,B = pred(nzr_p,nzr_n,th)
                    AI.append(A)
                    BI.append(B)
                AI = np.array(AI)
                BI = np.array(BI)
                return AI,BI

            A1,B1 = pred_roc(unz_px,unz_nx)
            A2,B2 = pred_roc(lnz_px,lnz_nx)
            A3,B3 = pred_roc(unz_py,unz_ny)
            A4,B4 = pred_roc(lnz_py,lnz_ny)
            A11,B11 = pred_roc(unzs_px,unzs_nx)
            A22,B22 = pred_roc(lnzs_px,lnzs_nx)
            A33,B33 = pred_roc(unzs_py,unzs_ny)
            A44,B44 = pred_roc(lnzs_py,lnzs_ny)
            
            all_combinations = list(itertools.product(range(align_px), repeat=8))
            tprs = []
            fprs = []
            tpri=-1
            for i,j,k,l,m,n,o,p in all_combinations:
                a = np.floor((A1[i]+A2[j]+A3[k]+A4[l]+A11[m]+A22[n]+A33[o]+A44[p])/8)
                b = np.floor((B1[i]+B2[j]+B3[k]+B4[l]+B11[m]+B22[n]+B33[o]+B44[p])/8)
                _,tprr,fpri =  pr(a, b)
                if tprr >= tpri:
                    tpri = tprr
                    tprs.append(tpri)
                    fprs.append(fpri)
            self.tpr_list = tprs
            self.fpr_list = fprs

class DMF_A_test():
    def __init__(self,model,testp_dir,testn_dir,plot=False):
        ## path = path of the file
        ## ni = quantity of sampling 
        def read(path):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            datat = []
            for path in PATH:
                img = cv2.imread(path)
                resize_img = cv2.resize(img,(200,200))#resize
                resize_img = rotate_img(resize_img, angle=model.ang)#rotate
                gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)#gray
                blur = cv2.medianBlur(gray,5)#blur
                canny_img = cv2.Canny(blur, 50, 100)#canny
                serise = np.sum(canny_img,axis=0)
                serise_t = np.sum(canny_img.T,axis=0)
                data.append([serise])
                datat.append([serise_t])
            data=np.array(data)
            datat=np.array(datat)
            return data,datat
       
            
        align_px,align_py = read(testp_dir, model.ni)
        align_nx,align_ny = read(testn_dir, model.ni)
        def compressed(data):
            unz=[]
            lnz=[]
            for i in range(len(data)):
                x=data[i]
                upper = model.bound[0]
                lower = model.bound[1]
                upzs=0
                lowzs=0
                for j in range(len(x)):
                    if x[j] > upper[j]:
                        upzs+=1
                    elif x[j] < lower[j]:
                        lowzs+=1
                unz.append(upzs/len(x))
                lnz.append(lowzs/len(x))
            return unz,lnz
        def pred(P,N,threshold):
            a = []
            b = []
            for i in P:
                if i > threshold:
                    a.append(0)
                elif i <= threshold:
                    a.append(1)
            for j in N:
                if j > threshold:
                    b.append(0)
                elif j <= threshold:
                    b.append(1)
            a = np.array(a)
            b = np.array(b)
            return a,b
        
        def pr(a,b):#a是目標，b是非目標
            tp = sum(a)
            fn = sum([abs(k-1) for k in a])
            tn = sum([abs(l-1) for l in b])
            fp = sum(b)
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            accu = (tp+tn)/(tp+tn+fp+fn)
            youden = tpr - fpr
            return accu,youden,tpr,fpr
        
        unz_px,lnz_px = compressed(align_px)
        unz_nx,lnz_nx = compressed(align_nx)
        unz_py,lnz_py = compressed(align_py)
        unz_ny,lnz_ny = compressed(align_ny)

        
        crupx,crunx = pred(unz_px,unz_nx,model.threshold[0])
        crlpx,crlnx = pred(lnz_px,lnz_nx,model.threshold[1])
        crupy,cruny = pred(unz_py,unz_ny,model.threshold[2])
        crlpy,crlny = pred(lnz_py,lnz_ny,model.threshold[3])

        
        POSITIVE = np.where(np.mean([crupx,crlpx,crupy,crlpy],axis=0)<1,0,1)
        NEGATIVE = np.where(np.mean([crunx,crlnx,cruny,crlny],axis=0)<1,0,1)
        reallabel_p = np.ones(len(POSITIVE))
        reallabel_n = np.zeros(len(NEGATIVE))
        classfy = {"real":np.concatenate((reallabel_p,reallabel_n)),"predict":np.concatenate((POSITIVE,NEGATIVE))}
        
        accuracy,youden,tpr,fpr=pr(POSITIVE, NEGATIVE)
        
        self.result = classfy
        self.accuracy = accuracy
        self.youden = youden
        self.tpr = tpr
        self.fpr = fpr
        
        if plot == True:
            def ulimit(array,how):
                for i in range(len(array)):
                    if sum(array[i])==how:
                        ii=i+1
                        return ii
            def pred_roc(nzr_p,nzr_n):
                AI=[]
                BI=[]
                for th in sorted(set(nzr_p+nzr_n)):
                    A,B = pred(nzr_p,nzr_n,th)
                    AI.append(A)
                    BI.append(B)
                AI = np.array(AI)
                BI = np.array(BI)
                return AI,BI

            A1,B1 = pred_roc(unz_px,unz_nx)
            A2,B2 = pred_roc(lnz_px,lnz_nx)
            A3,B3 = pred_roc(unz_py,unz_ny)
            A4,B4 = pred_roc(lnz_py,lnz_ny)

            
            all_combinations = list(itertools.product(range(align_px), repeat=4))
            tprs = []
            fprs = []
            tpri=-1
            for i,j,k,l,m,n,o,p in all_combinations:
                a = np.floor((A1[i]+A2[j]+A3[k]+A4[l])/4)
                b = np.floor((B1[i]+B2[j]+B3[k]+B4[l])/4)
                _,tprr,fpri =  pr(a, b)
                if tprr >= tpri:
                    tpri = tprr
                    tprs.append(tpri)
                    fprs.append(fpri)
            self.tpr_list = tprs
            self.fpr_list = fprs
