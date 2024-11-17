# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:26:19 2024

@author: USER
"""
#pip install cv2
#pip install numpy
#pip install os
#pip install math
#pip install scipy
#pip install sklearn
#pip install itertools

import cv2
import os
import numpy as np
import math
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.spatial.distance import cdist
from SSG import AC,aligns
from imgRotate import rotate_img,corp_canny
import itertools

""""""""""    DKF    """""""""
class DKF():
    ### P_dir is the directory of Positive dataset
    ### N_dir is the directory of Negative dataset
    ## ang = rotation angle
    def __init__(self,P_dir,N_dir,ni,ang=0):
        """ Step 1 """
        """ Read the dataset """
        ### path = the directory of dataset
        def read(path,ang):
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
                resize_img = rotate_img(resize_img, ang)#旋轉圖片
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
        
        
        a,b = read(P_dir,ang)## Positive dataset: a is x-axis projectin & b is y-axis projectin
        o,g = read(N_dir,ang)## Negative dataset: o is x-axis projectin & g is y-axis projectin
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        mean_x=AC(a).reshape(-1,1)### Computes the mean of x-axis projectin(positive)
        mean_y=AC(b).reshape(-1,1)### Computes the mean of y-axis projectin(positive)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 2 """
        """ FRK """
        ## path = path of the file
        ## ni = quantity of sampling 
        def read_frk(path,ni,ang=ang):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            for path in PATH:
                try:
                    img = cv2.imread(path)
                    resize_img = cv2.resize(img,(200,200))#resize
                    resize_img = rotate_img(resize_img, ang)#rotation image
                    cut_goal = corp_canny(resize_img,upper=150,lower=50)
                    goal_canny = cv2.resize(cut_goal,(200,200))
                    mat1 = aligns(goal_canny,mean_y,"y")## image align to mean of y-axis
                    mat2 = aligns(mat1,mean_x,"x")## mat1 align to mean of x-axis
                    nonzero_points = np.column_stack(np.where(mat2 >= 230))
                    num_samples = ni
                    sample_indices = np.linspace(0, len(nonzero_points) - 1, num_samples, dtype=int)
                    ti = nonzero_points[sample_indices]
                    coord_x = []
                    values = []
                    for point in ti:
                        x, y = point
                        value = mat2[x, y]
                        coord_x.append([x,y])
                        values.append(value)
                    coord_x = np.array(coord_x)
                    values = np.array(values)
                    length_scale,rank,noise_level = 0.1,5,0.01
                    distances = cdist(coord_x, coord_x)
                    correlation_matrix = np.exp(-distances / length_scale)
                    u, s, vh = np.linalg.svd(correlation_matrix)
                    u_r = u[:, :rank]
                    s_r = np.diag(s[:rank])
                    vh_r = vh[:rank, :]
                    corr_matrix_approx_g = np.dot(u_r, np.dot(s_r, vh_r))
                    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                    gpr.fit(coord_x, values)
                    mean_g, std = gpr.predict(coord_x, return_std=True)
                    mean_frk_g = np.dot(u_r, np.dot(np.linalg.inv(s_r), np.dot(vh_r, mean_g.reshape(-1, 1))))
                    data.append(np.concatenate((mean_frk_g,corr_matrix_approx_g.flatten('C').reshape(-1,1)),axis = 0))
                except:
                    data.append(np.zeros(len(np.concatenate((mean_frk_g,corr_matrix_approx_g.flatten('C').reshape(-1,1)),axis = 0))).reshape(-1,1))
                    pass
            return data
        
        
        align_p = read_frk(P_dir, ni)
        align_n = read_frk(N_dir, ni)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 3 """
        mean_p = np.mean(align_p, axis=0).reshape(-1,1)
        cov = []
        for i in range(len(mean_p)):
            cov.append([math.sqrt(np.var(align_p[:,i,0]))])
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 4 """
        """ Create the envelope """
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
                counter=[]## to check the index of peaks
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
        k1,ubx,nzn_u_p,nzn_u_n,nzr_u_p,nzr_u_n = boundary(align_p, align_n, mean_p, cov, judge="upper")
        k11,lbx,nzn_l_p,nzn_l_n,nzr_l_p,nzr_l_n = boundary(align_p, align_n, mean_p, cov, judge="lower")
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
        
        delindex = list(set(out_of(nzr_l_n, nzr_l_p)+out_of(nzr_u_n, nzr_u_p)))
        notdel = [index for index, element in enumerate(align_n) if index not in delindex]
        """"""""""""""""""""""""""""""""""""""
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
        
        # The "smelt" step of DKF
        def smelt(Pnzn,Nnzn,notdelet,ul,qrange=np.round(np.arange(0,5.1,0.01),1)):
            data_p = align_p
            data_n = align_n
            judge = ul
            if ul == "upper":
                bound = ubx
            elif ul == "lower":
                bound = lbx
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
        
        snzr_u_p,snzr_u_n,notdel,peaks_u = smelt(nzn_u_p, nzn_u_n, notdel, "upper")
        snzr_l_p,snzr_l_n,notdel,peaks_l = smelt(nzn_l_p, nzn_l_n, notdel, "lower")
        """"""""""""""""""""""""""""""""""""""
        """ Step 3 & Step 7 """
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
        
        
        A1,B1,t1 = DTfilte(nzr_u_p, nzr_u_n)
        A2,B2,t2 = DTfilte(nzr_l_p, nzr_l_n)
        SA1,SB1,t11 = DTfilte(snzr_u_p, snzr_u_n)
        SA2,SB2,t22 = DTfilte(snzr_l_p, snzr_l_n)

        
        POSITIVE = np.where(np.mean([A1,A2,SA1,SA2],axis=0)<1,0,1)
        NEGATIVE = np.where(np.mean([B1,B2,SB1,SB2],axis=0)<1,0,1)
        reallabel_p = np.ones(len(POSITIVE))
        reallabel_n = np.zeros(len(NEGATIVE))
        classfy = {"real":np.concatenate((reallabel_p,reallabel_n)),"predict":np.concatenate((POSITIVE,NEGATIVE))}
        
        accuracy,youden,tpr,fpr=pr(POSITIVE, NEGATIVE)
        
            
        self.meanx = mean_x
        self.meany = mean_y
        self.k = {0:k1,1:k11}
        self.bound = {0:ubx,1:lbx}
        self.accuracy = accuracy
        self.youden = youden
        self.tpr = tpr
        self.fpr = fpr
        self.result = classfy
        self.nzr = {0:nzr_u_p,1:nzr_u_n,2:nzr_l_p,3:nzr_l_n}
        self.subnzr = {0:snzr_u_p,1:snzr_u_n,2:snzr_l_p,3:snzr_l_n}
        self.ang = ang
        self.ni = ni
        self.peaks = {0:peaks_u,1:peaks_l}
        self.threshold = {0:t1,1:t2,2:t11,3:t22}

""""""""""    DKF_test   """""""""
class DKF_test():
    def __init__(self,model,testp_dir,testn_dir,plot=False):
        ## path = path of the file
        ## ni = quantity of sampling 
        def read_frk(path,ni,ang=model.ang):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            for path in PATH:
                try:
                    img = cv2.imread(path)
                    resize_img = cv2.resize(img,(200,200))#resize
                    resize_img = rotate_img(resize_img, ang)#rotation image
                    cut_goal = corp_canny(resize_img,upper=150,lower=50)
                    goal_canny = cv2.resize(cut_goal,(200,200))
                    mat1 = aligns(goal_canny,model.meany,"y")## image align to mean of y-axis
                    mat2 = aligns(mat1,model.meanx,"x")## mat1 align to mean of x-axis
                    nonzero_points = np.column_stack(np.where(mat2 >= 230))
                    num_samples = ni
                    sample_indices = np.linspace(0, len(nonzero_points) - 1, num_samples, dtype=int)
                    ti = nonzero_points[sample_indices]
                    coord_x = []
                    values = []
                    for point in ti:
                        x, y = point
                        value = mat2[x, y]
                        coord_x.append([x,y])
                        values.append(value)
                    coord_x = np.array(coord_x)
                    values = np.array(values)
                    length_scale,rank,noise_level = 0.1,5,0.01
                    distances = cdist(coord_x, coord_x)
                    correlation_matrix = np.exp(-distances / length_scale)
                    u, s, vh = np.linalg.svd(correlation_matrix)
                    u_r = u[:, :rank]
                    s_r = np.diag(s[:rank])
                    vh_r = vh[:rank, :]
                    corr_matrix_approx_g = np.dot(u_r, np.dot(s_r, vh_r))
                    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                    gpr.fit(coord_x, values)
                    mean_g, std = gpr.predict(coord_x, return_std=True)
                    mean_frk_g = np.dot(u_r, np.dot(np.linalg.inv(s_r), np.dot(vh_r, mean_g.reshape(-1, 1))))
                    data.append(np.concatenate((mean_frk_g,corr_matrix_approx_g.flatten('C').reshape(-1,1)),axis = 0))
                except:
                    data.append(np.zeros(len(np.concatenate((mean_frk_g,corr_matrix_approx_g.flatten('C').reshape(-1,1)),axis = 0))).reshape(-1,1))
                    pass
            return data
       
            
        align_p = read_frk(testp_dir, model.ni)
        align_n = read_frk(testn_dir, model.ni)
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
        
        unz_p,lnz_p = compressed(align_p)
        unz_n,lnz_n = compressed(align_n)
        unzs_p,lnzs_p = compressed_s(align_p,model.peaks)
        unzs_n,lnzs_n = compressed_s(align_n,model.peaks)
        
        crup,crun = pred(unz_p,unz_n,model.threshold[0])
        crlp,crln = pred(lnz_p,lnz_n,model.threshold[1])
        crups,cruns = pred(unzs_p,unzs_n,model.threshold[2])
        crlps,crlns = pred(lnzs_p,lnzs_n,model.threshold[3])
        
        POSITIVE = np.where(np.mean([crup,crlp,crups,crlps],axis=0)<1,0,1)
        NEGATIVE = np.where(np.mean([crun,crln,cruns,crlns],axis=0)<1,0,1)
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

            A1,B1 = pred_roc(unz_p,unz_n)
            A2,B2 = pred_roc(lnz_p,lnz_n)
            A11,B11 = pred_roc(unzs_p,unzs_n)
            A22,B22 = pred_roc(lnzs_p,lnzs_n)
            
            ntype = 4
            all_combinations = list(itertools.product(range(align_p), repeat=ntype))
            tprs = []
            fprs = []
            tpri=-1
            for i,j,k,l,m,n,o,p in all_combinations:
                a = np.floor((A1[i]+A2[j]+A11[m]+A22[n])/ntype)
                b = np.floor((B1[i]+B2[j]+B11[m]+B22[n])/ntype)
                _,tprr,fpri =  pr(a, b)
                if tprr >= tpri:
                    tpri = tprr
                    tprs.append(tpri)
                    fprs.append(fpri)
                    
            self.tpr_list = tprs
            self.fpr_list = fprs

        

""""""""""    DKF_choosen    """""""""
class DKF_choosen():
    ### P_dir is the directory of Positive dataset
    ### N_dir is the directory of Negative dataset
    ## ang = rotation angle
    def __init__(self,P_dir,N_dir,ni,ang=0,stop = 90):
        """ Step 1 """
        """ Read the dataset """
        ### path = the directory of dataset
        def read(path,ang):
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
                resize_img = rotate_img(resize_img, ang)#旋轉圖片
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
        
        
        a,b = read(P_dir,ang)## Positive dataset: a is x-axis projectin & b is y-axis projectin
        o,g = read(N_dir,ang)## Negative dataset: o is x-axis projectin & g is y-axis projectin
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        mean_x=AC(a).reshape(-1,1)### Computes the mean of x-axis projectin(positive)
        mean_y=AC(b).reshape(-1,1)### Computes the mean of y-axis projectin(positive)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Step 2 """
        """ FRK """
        ## path = path of the file
        ## ni = quantity of sampling
        
        def read_frk(path,ni,ang=ang):
            myPath = path
            otherList=os.walk(myPath)
            PATH = []
            for root, dirs, files in otherList:
                if root!=myPath:
                    for i in files:
                        PATH.append(root+str("/")+str(i))
            data = []
            for path in PATH:
                try:
                    img = cv2.imread(path)
                    resize_img = cv2.resize(img,(200,200))#resize
                    resize_img = rotate_img(resize_img, ang)#rotation image
                    cut_goal = corp_canny(resize_img,upper=150,lower=50)
                    goal_canny = cv2.resize(cut_goal,(200,200))
                    mat1 = aligns(goal_canny,mean_y,"y")## image align to mean of y-axis
                    mat2 = aligns(mat1,mean_x,"x")## mat1 align to mean of x-axis
                    nonzero_points = np.column_stack(np.where(mat2 >= 230))
                    if ni == 10:
                        num_samples = ni
                        sample_indices = np.linspace(0, len(nonzero_points) - 1, num_samples, dtype=int)
                        ti = nonzero_points[sample_indices]
                    else:
                        num_samples = ni
                        sample_indices = np.linspace(0, len(nonzero_points) - 1, num_samples-10, dtype=int)[choosen]
                        
                        allind = np.array(range(len(nonzero_points)))
                        selected_numbers = allind[~np.isin(allind, sample_indices)]
                        selected_numbers = np.random.choice(selected_numbers, ni-len(choosen), replace=False)
                        ti = nonzero_points[np.union1d(sample_indices,selected_numbers)]
                    coord_x = []
                    values = []
                    for point in ti:
                        x, y = point
                        value = mat2[x, y]
                        coord_x.append([x,y])
                        values.append(value)
                    coord_x = np.array(coord_x)
                    values = np.array(values)
                    length_scale,rank,noise_level = 0.1,5,0.01
                    distances = cdist(coord_x, coord_x)
                    correlation_matrix = np.exp(-distances / length_scale)
                    u, s, vh = np.linalg.svd(correlation_matrix)
                    u_r = u[:, :rank]
                    s_r = np.diag(s[:rank])
                    vh_r = vh[:rank, :]
                    corr_matrix_approx_g = np.dot(u_r, np.dot(s_r, vh_r))
                    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
                    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                    gpr.fit(coord_x, values)
                    mean_g, std = gpr.predict(coord_x, return_std=True)
                    mean_frk_g = np.dot(u_r, np.dot(np.linalg.inv(s_r), np.dot(vh_r, mean_g.reshape(-1, 1))))
                    data.append(np.concatenate((mean_frk_g,corr_matrix_approx_g.flatten('C').reshape(-1,1)),axis = 0))
                except:
                    data.append(np.zeros(len(np.concatenate((mean_frk_g,corr_matrix_approx_g.flatten('C').reshape(-1,1)),axis = 0))).reshape(-1,1))
                    pass
            return data
        
        accuracy_list=[]
        youden_list=[]
        tpr_list=[]
        fpr_list=[]
        choosen = []
        while ni <= stop :  
            align_p = read_frk(P_dir, ni)
            align_n = read_frk(N_dir, ni)
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            """ Step 3 """
            mean_p = np.mean(align_p, axis=0).reshape(-1,1)
            cov = []
            for i in range(len(mean_p)):
                cov.append([math.sqrt(np.var(align_p[:,i,0]))])
            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            """ Step 4 """
            """ Create the envelope """
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
                    counter=[]## to check the index of peaks
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
            k1,ubx,nzn_u_p,nzn_u_n,nzr_u_p,nzr_u_n = boundary(align_p, align_n, mean_p, cov, judge="upper")
            k11,lbx,nzn_l_p,nzn_l_n,nzr_l_p,nzr_l_n = boundary(align_p, align_n, mean_p, cov, judge="lower")
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
            
            delindex = list(set(out_of(nzr_l_n, nzr_l_p)+out_of(nzr_u_n, nzr_u_p)))
            notdel = [index for index, element in enumerate(align_n) if index not in delindex]
            """"""""""""""""""""""""""""""""""""""
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
            
            # The "smelt" step of DKF
            def smelt(Pnzn,Nnzn,notdelet,ul,qrange=np.round(np.arange(0,5.1,0.01),1)):
                data_p = align_p
                data_n = align_n
                judge = ul
                if ul == "upper":
                    bound = ubx
                elif ul == "lower":
                    bound = lbx
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
            
            snzr_u_p,snzr_u_n,notdel,peaks_u = smelt(nzn_u_p, nzn_u_n, notdel, "upper")
            snzr_l_p,snzr_l_n,notdel,peaks_l = smelt(nzn_l_p, nzn_l_n, notdel, "lower")
            choosen = np.union1d(peaks_u[np.where(peaks_u<ni)],peaks_l[np.where(peaks_l<ni)])
            """"""""""""""""""""""""""""""""""""""
            """ Step 3 & Step 7 """
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
            
            
            A1,B1,t1 = DTfilte(nzr_u_p, nzr_u_n)
            A2,B2,t2 = DTfilte(nzr_l_p, nzr_l_n)
            SA1,SB1,t11 = DTfilte(snzr_u_p, snzr_u_n)
            SA2,SB2,t22 = DTfilte(snzr_l_p, snzr_l_n)
    
            
            POSITIVE = np.where(np.mean([A1,A2,SA1,SA2],axis=0)<1,0,1)
            NEGATIVE = np.where(np.mean([B1,B2,SB1,SB2],axis=0)<1,0,1)
            reallabel_p = np.ones(len(POSITIVE))
            reallabel_n = np.zeros(len(NEGATIVE))
            classfy = {"real":np.concatenate((reallabel_p,reallabel_n)),"predict":np.concatenate((POSITIVE,NEGATIVE))}
            accuracy,youden,tpr,fpr=pr(POSITIVE, NEGATIVE)
            accuracy_list.append(accuracy)
            youden_list.append(youden)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
            
        self.meanx = mean_x
        self.meany = mean_y
        self.k = {0:k1,1:k11}
        self.bound = {0:ubx,1:lbx}
        self.accuracy = accuracy_list
        self.youden = youden_list
        self.tpr = tpr_list
        self.fpr = fpr_list
        self.result = classfy
        self.nzr = {0:nzr_u_p,1:nzr_u_n,2:nzr_l_p,3:nzr_l_n}
        self.subnzr = {0:snzr_u_p,1:snzr_u_n,2:snzr_l_p,3:snzr_l_n}
        self.ang = ang
        self.ni = ni
        self.peaks = {0:peaks_u,1:peaks_l}
        self.threshold = {0:t1,1:t2,2:t11,3:t22}