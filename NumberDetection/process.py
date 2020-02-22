from __future__ import print_function
#import potrebnih biblioteka
import cv2

from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from joblib import dump, load

import numpy as np
import matplotlib.pylab as plt

nbins = 8 # broj binova
cell_size = (6, 6) # broj piksela po celiji
block_size = (2, 2) # broj celija po bloku
size=(30,30)

from keras.datasets import mnist
(trainX, trainy), (testX, testy) = mnist.load_data()

def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=3)
def preprocessImage(img):
    img = image_gray( img)
    img=255-img
    img = dilate(img) 
    imbin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    img1, contours, hierarchy = cv2.findContours(imbin,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = []
    regions=[]
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if w<30 or h<50 or w > 400 or h > 400:
                continue
        for inside in contours: 
                 x1,y1,w1,h1 = cv2.boundingRect(inside)
                 if w1<30 or h1<50 or w1 > 400:
                     continue
                 if x1>x and x+w>x1+w1:
                     cont.append(x1)
                     break
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if w<30 or h<50 or w > 400 or h > 400:
                continue
        if x not in cont :            
            region=imbin[y:y+h+1,x:x+w+1]
            regions.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 255), 4)
    
    return regions

def trainSVM(pos_features):
    x_train = reshape_data(pos_features)
    y = np.array(trainy[0:40000])
    print(x_train.shape,y.shape)
    
    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y)
    y_train_pred = clf_svm.predict(x_train)    
    print("Train accuracy: ", accuracy_score(y, y_train_pred))
    dump(clf_svm, 'digitDetection.joblib')  
    return
def preprocessDataSVM():
    pos_features=[]
    for x in trainX[0:40000] :
        img=x
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
        pos_features.append(hog.compute(img))
    pos_features = np.array(pos_features)
    return pos_features
def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized


def load_image_to_hsv(path):
    return cv2.cvtColor(path,cv2.COLOR_BGR2HSV)
def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count
def testAcc():
    m = load('digitDetection.joblib')    
    #Testiranje na slikama 98%
    pos_features=[]
    for x in  testX :
            img=x
            hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                          img.shape[0] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
            pos_features.append(hog.compute(img))
    pos_features = np.array(pos_features)
    x_test = reshape_data(pos_features)
    y = np.array(testy)
    y_train_pred = m.predict(x_test)    
    print("Test accuracy: ", accuracy_score(y, y_train_pred))

#trainSVM(preprocessDataSVM())
#testAcc()
m = load('digitDetection.joblib')  
zbir = 0
cap = cv2.VideoCapture('try3.mp4')
prev = -1
a=[]
i=0
zbir=0
pred = []
while(True):   
        i+=1
        ret, frame = cap.read()
        img = image_gray(frame)
        img=255-img
        img = dilate(img) 
        imbin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        img1, contours, hierarchy = cv2.findContours(imbin,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maximum=0
        for contour in contours: 
            x,y,w,h = cv2.boundingRect(contour)  
            if maximum<w*h and w<300:
                maximum =w*h
        for contour in contours: 
            x,y,w,h = cv2.boundingRect(contour)            
            if maximum==w*h : 
                img = image_gray(frame)
                img = 255-img  
                img = img[y:y+h,x:x+w]
                hsv = load_image_to_hsv(frame);
                sensitivity = 50;
                lower = np.array([30 - sensitivity, 100+sensitivity, 50])  
                upper = np.array([30 + sensitivity, 255+sensitivity, 255])
                mask = cv2.inRange(hsv,lower,upper);
                imbin = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
                img1, contours, hierarchy = cv2.findContours(imbin,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                neg=False
                if len(contours)>20 :
                    neg=True
                img = cv2.resize(img,(28,28))
                hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                              img.shape[0] // cell_size[0] * cell_size[0]),
                                    _blockSize=(block_size[1] * cell_size[1],
                                                block_size[0] * cell_size[0]),
                                    _blockStride=(cell_size[1], cell_size[0]),
                                    _cellSize=(cell_size[1], cell_size[0]),
                                    _nbins=nbins)
                pos_features=hog.compute(img)
                pos_features = np.array(pos_features).reshape(1,-1)
                m.predict(pos_features)
                res=(m.predict(pos_features))
                pred.append(res)
                if len(pred)>15:
                    pred.pop(0)
                if countX(pred,res)==15 and res not in a:                        
                    a.append(res)
                    if neg==True:
                        zbir-=res[0]
                    else:
                        zbir+=res[0]
                    print(res[0],neg)
                    pred.clear()
               
        if cv2.waitKey(1) == 27:
            break
cap.release()
