# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:33:22 2024

@author: USER
"""
#pip install numpy
#pip install matplotlib
#pip install sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

"""""""""""""""""""""""""""""" "Figure 3.2 & 3.5 & 3.7 & 4.18"  """"""""""""""""""""""""""""""""""""  
## fpr = false positive rate
## tpr = true positive rate
## color = the color of the curve
## pcolor = the color of the max youden 
## name = the name of the method to show
## linestyle = the type of the line
## yd = displayed value is youden or auc
def plotroc(fpr,tpr,color="b",pcolor="r",name="none",linestyle="-",yd=True,nif=True):
    if yd == True:
        n = max(np.array(tpr)-np.array(fpr))
        q = "Youden"
    elif yd == False:
        n = auc(fpr,tpr)
        q = "AUC"
    if nif == True:
        plt.plot(fpr,tpr,label = "%s(%s) = %0.3f" %(q,name,n),color = color,linestyle = linestyle)
        youdeninde = np.argmax(np.array(tpr)-np.array(fpr))
        plt.plot(fpr[youdeninde],tpr[youdeninde],".",color=pcolor)
        plt.plot([fpr[youdeninde],fpr[youdeninde]],[fpr[youdeninde],tpr[youdeninde]],":",color=pcolor)
    elif nif == False:
        plt.plot(fpr,tpr,label = "%s(%s) = %0.3f" %(q,name,n),color = color,linestyle = linestyle)
        youdeninde = np.argmax(np.array(tpr)-np.array(fpr))
        plt.plot(fpr[youdeninde],tpr[youdeninde],".",color=pcolor)
"""""""""""""""""""""""""""""" "Figure 3.3 & 3.6" """"""""""""""""""""""""""""""""""""
## nzr = [non-zero ratio of upper x-axis, lower x-axis, upper y-axis, lower y-axis]
## th = [threshold of upper x-axis, lower x-axis, upper y-axis, lower y-axis]
def boxplot4(nzr,th,save_dir):
    # build 4 figure plane
    plt.figure(figsize=(10, 8))
    
    # upper x-axis projection
    plt.subplot(2, 2, 1)
    plt.boxplot([nzr[0],nzr[1]],['Positive','Negative'],labels=['Positive','Negative'],patch_artist = True)#,rohs_ndistmin
    plt.axhline(y=th[0],color = "r",linestyle = "--")
    #plt.xlabel("(a)")
    plt.title("(a)")
    
    # lower x-axis projection
    plt.subplot(2, 2, 2)
    plt.boxplot([nzr[2],nzr[3]],['Positive','Negative'],labels=['Positive','Negative'],patch_artist = True)#,rohs_ndistmin
    plt.axhline(y=th[1],color = "r",linestyle = "--")
    plt.title("(b)")
    
    # upper y-axis projection
    plt.subplot(2, 2, 3)
    plt.boxplot([nzr[4],nzr[5]],['Positive','Negative'],labels=['Positive','Negative'],patch_artist = True)#,rohs_ndistmin
    plt.axhline(y=th[2],color = "r",linestyle = "--")
    plt.title("(c)")
    
    # lower y-axis projection
    plt.subplot(2, 2, 4)
    plt.boxplot([nzr[6],nzr[7]],['Positive','Negative'],labels=['Positive','Negative'],patch_artist = True)#,rohs_ndistmin
    plt.axhline(y=th[3],color = "r",linestyle = "--")
    plt.title("(d)")
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.savefig(save_dir,dpi=500)
"""""""""""""""""""""""""""""" "Figure 3.4" """"""""""""""""""""""""""""""""""""  
def plot_tau(tau,q,peaks,save_dir):    
    plt.plot(tau)
    plt.plot(peaks,tau[peaks],"x",color = "g" )
    plt.hlines(q,0,199,"red",linewidth = 1)
    #plt.title("RoHS y lower bound (others/rohs)")
    plt.ylabel(r"$\tau_t$")
    plt.xlabel("t")
    plt.savefig(save_dir,dpi=500)
"""""""""""""""""""""""""""""" "Figure 4.6 & 4.15" """"""""""""""""""""""""""""""""""""  
def ROC_compare(fpr_OCR,tpr_OCR,fpr_CNN,tpr_CNN,fpr_DMFA,tpr_DMFA,fpr_DMFS,tpr_DMFS,fpr_DKF,tpr_DKF,save_dir):
    auc_OCR=auc(fpr_OCR,tpr_OCR)
    auc_CNN=auc(fpr_CNN,tpr_CNN)
    auc_DMFA=auc(fpr_DMFA,tpr_DMFA)
    auc_DMFS=auc(fpr_DMFS,tpr_DMFS)
    auc_FRK=auc(fpr_DKF,tpr_DKF)
    plt.title('Receiver Operating Characteristic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel("TPR",fontsize = 14)
    plt.xlabel("FPR",fontsize = 14)
    plt.plot(fpr_OCR,tpr_OCR,label = "AUC(OCR) = %0.2f" % auc_OCR,color = "brown")
    plt.plot(fpr_CNN,tpr_CNN,label = "AUC(CNN) = %0.2f" % auc_CNN,color = 'g')
    plt.plot(fpr_DMFA,tpr_DMFA,label = "AUC(DMF-A) = %0.2f" % auc_DMFA,color = 'darkgoldenrod',linestyle = '--')
    plt.plot(fpr_DMFS,tpr_DMFS,label = "AUC(DMF-S) = %0.2f" % auc_DMFS,color = "darkorange")
    plt.plot(fpr_DKF,tpr_DKF,label = "AUC(DKF) = %0.2f" % auc_FRK,color = "b")
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.savefig(save_dir,dpi=500)
"""""""""""""""""""""""""""""" "Figure 4.16" """"""""""""""""""""""""""""""""""""   
def ROC_pp(fpr_DMFA, tpr_DMFA,fpr_DMFS, tpr_DMFS,fpr_FRK, tpr_FRK,save_dir):
    y1 = np.argmax(np.array(tpr_DMFA)-np.array(fpr_DMFA))
    y2 = np.argmax(np.array(tpr_DMFS)-np.array(fpr_DMFS))
    plt.figure(figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.xlim(0, 0.2)
    plt.ylim(0, 1.1)
    plt.ylabel("TPR",fontsize = 14)
    plt.xlabel("FPR",fontsize = 14)
    plt.plot([0, 1], [1, 1],color ='gray',linestyle =':')
    plotroc(fpr_DMFA, tpr_DMFA,'darkgoldenrod',"hotpink","DMF-A","--")
    plotroc(fpr_DMFS, tpr_DMFS,"darkorange","m","DMF-S")
    plotroc(fpr_FRK, tpr_FRK,"b","c","DKF")
    plt.plot(fpr_DMFA[y1],tpr_DMFA[y1],".",color = 'hotpink')
    plt.plot(fpr_DMFS[y2],tpr_DMFS[y2],".",color = 'm')
    plt.legend(loc = 'lower right',fontsize=7)
    plt.plot([0, 1], [0, 1],'r--')
    plt.savefig(save_dir,dpi=500)
"""""""""""""""""""""""""""""" "Figure 5.2" """"""""""""""""""""""""""""""""""""  
def youden_deg(fpr,tpr,save_dir):
    LINE = np.array(tpr)-np.array(fpr)
    LINE2 = np.array(fpr)-np.array(tpr)
    
    plt.plot(LINE,label =' Youden')
    plt.plot(LINE2,label = ' -Youden')
    plt.legend(loc = 'best')
    plt.ylabel(" Youden",fontsize = 12)
    plt.xlabel("H",fontsize = 12)
    plt.plot(np.argmax(LINE),max(LINE),".")
    plt.plot(np.argmin(LINE2),min(LINE2),".")
    plt.text(np.argmax(LINE)+0.5,max(LINE),"%0.3f"%(max(LINE)))
    plt.text(np.argmin(LINE2)+0.5,min(LINE2)-0.03,"%0.3f"%(min(LINE2)))
    plt.plot([-0.5, 20.5], [0, 0],'r:')
    #plt.show()
    plt.savefig(save_dir,dpi=500)

