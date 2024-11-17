#pip install cv2
#pip install os
#pip install numpy
#pip install keras
#pip install random
#pip install tensorflow
#pip install sklearn


import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
import random
from keras.optimizers import Adam
import tensorflow as tf
from sklearn import metrics

#path = the directory of  dataset
#label = the label of the dataset
def read(path,label):
    myPath = path
    otherList=os.walk(myPath)
    PATH = []
    for root, dirs, files in otherList:
        if root!=myPath:
            for i in files:
                PATH.append(root+str("/")+str(i))
    label = []
    data = []
    for path in PATH:
        img = cv2.imdecode(np.fromfile(file="{}".format(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        resize_img = cv2.resize(img,(200,200))
        goal_gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)#灰階化
        blur_goal = cv2.medianBlur(goal_gray,5)#模糊降躁
        img_canny = cv2.Canny(blur_goal, 50, 100)#邊緣偵測
        blur_goal = cv2.adaptiveThreshold(blur_goal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY, 11, 2)
        img_canny = cv2.resize(blur_goal,(200,200))#調整圖片大小
        data.append(img_canny)
        label.append(label)
    data=np.array(data)
    label=np.array(label)
    return data,label

### P_dir is the directory of Positive dataset
### N_dir is the directory of Negative dataset
class CNN():
    def __init__(self,P_path,N_path):
        data,plabel = read(P_path,1)
        ott,nlabel = read(N_path,0)
        
        train_1 = data
        lab_1 = plabel
        train_0 = ott
        lab_0 = nlabel
        
        # Normalize the data
        train_1 = np.array(train_1) / 255
        train_0 = np.array(train_0) / 255
        
        train_1.reshape(-1, 200,200, 1)
        lab_1 = np.array(lab_1)
        
        train_0.reshape(-1, 200,200, 1)
        lab_0 = np.array(lab_0)
        

        train_goal = train_1
        train_other = train_0
        trainset = np.concatenate((train_goal,train_other), axis=0)
        trainset=trainset.reshape(len(trainset), 200, 200,1)
        lab_goal = lab_1
        lab_other = lab_0
        traintargets = np.concatenate((lab_goal ,lab_other), axis=0)
        
        varidation = range((len(trainset)-1))
        vari = random.sample(varidation, round(1/3*len(trainset)))
        variset = trainset[vari]
        varitargets = traintargets[vari]
        
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # Randomly zoom image 
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip = True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
        
        datagen.fit(trainset)
        
        model = Sequential()
        model.add(Conv2D(16,kernel_size=(3, 3),padding="same", activation="relu", input_shape=(200,200,1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32,kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        model.add(Flatten())
        model.add(Dense(512,activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        
        # Compile the model
        opt = Adam(lr=0.000001)
        model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
        
        
        
        # Fit data to model
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)        
        history = model.fit(trainset,traintargets,epochs = 100 ,callbacks=[callback],validation_data = (variset, varitargets))     
        #acc = history.history['accuracy']
        #val_acc = history.history['val_accuracy']
        #loss = history.history['loss']
        #val_loss = history.history['val_loss']
        
        self.model = model
        self.history = history

### P_dir is the directory of Positive dataset
### N_dir is the directory of Negative dataset
def test_CNN(P_path,N_path,model):
    P_test,plabel = read(P_path)
    N_test,nlabel = read(N_path)
    cnn_model = model.model
    imge=np.concatenate((P_test ,N_test), axis=0)
    label=np.concatenate((plabel ,nlabel), axis=0)
    predic = cnn_model.predict(imge)
    fpr, tpr, thresholds = metrics.roc_curve(label, predic)
    return predic,fpr,tpr
    
#should install the executable files and language data: https://github.com/UB-Mannheim/tesseract/wiki
#pip install Pillow
#pip install pytesseract

from PIL import Image
from pytesseract import image_to_string

#path = the directory of image dataset
#language = "eng" or "chi_tra"
def OCR(path,language):
    myPath = path
    otherList=os.walk(myPath)
    PATH = []
    for root, dirs, files in otherList:
        if root!=myPath:
            for i in files:
                PATH.append(root+str("/")+str(i))
    text=[]
    for ipath in PATH:
        img = Image.open(ipath)
        text.append(image_to_string(img, lang=language))
    return text


