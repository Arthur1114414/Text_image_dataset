# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:11:47 2023

@author: USER
"""
#pip install cv2
#pip install numpy

import cv2
import numpy as np

def rotate_img(img,angle):
    (h, w, d) = img.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    colors = int(img[0,0,0])
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h),borderValue=(colors, colors, colors))
    
    return rotate_img

def corp_canny(img,width=200,hight=200,upper=50,lower=50):
    try:
        resize_img = cv2.resize(img,(width,hight))
        goal_gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)#灰階化
        blur_goal = cv2.medianBlur(goal_gray,5)#模糊降躁
        goal_canny = cv2.Canny(blur_goal, upper, lower)#邊緣偵測
        h,w = np.where(goal_canny==255)
        if len(h) != 0 or len(w)!=0:
            h_max = max(h)
            h_min = min(h)
            w_max = max(w)
            w_min = min(w)
        
            cut_canny = goal_canny[h_min:h_max,w_min:w_max]
            return cut_canny
        else:
            return goal_canny
    except:
        resize_img = cv2.resize(img,(width,hight))
        blur_goal = cv2.medianBlur(resize_img,5)#模糊降躁
        goal_canny = cv2.Canny(blur_goal, upper, lower)#邊緣偵測
        h,w = np.where(goal_canny==255)
        if len(h) != 0 or len(w)!=0:
            h_max = max(h)
            h_min = min(h)
            w_max = max(w)
            w_min = min(w)
        
            cut_canny = goal_canny[h_min:h_max,w_min:w_max]
            return cut_canny
        else:
            return goal_canny


def ftcolor(img,choose = "all",w=200,l=200):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img =  cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # green,black,whit
    c1 = np.array([35,43,46])
    c2 = np.array([77,255,255])

    b1 = np.array([0,0,0])
    b2 = np.array([180,255,46])

    w1 = np.array([0,0,221])
    w2 = np.array([180,30,255])
    
    mask1 = cv2.inRange(hsv_img,c1,c2)
    mask2 = cv2.inRange(hsv_img,b1,b2)
    mask3 = cv2.inRange(hsv_img,w1,w2)
    
    mask1 = cv2.resize(mask1,(w,l))
    mask2 = cv2.resize(mask2,(w,l))
    mask3 = cv2.resize(mask3,(w,l))
    if choose == "all":
        return mask1,mask2,mask3
    elif choose == "m1":
        return mask1
    elif choose == "m2":
        return mask2
    elif choose == "m3":
        return mask3