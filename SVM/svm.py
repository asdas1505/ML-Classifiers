#Your code goes here
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys
import pandas as pd

traindata = pd.read_csv("train.csv")

train_data = traindata.iloc[:,1:]

labels = traindata.iloc[:,0]

train_data['label'] = labels 


data = np.copy(train_data);

#print(data.shape)
x_train = data[:,0:data.shape[1]-1]
y_train = data[:,data.shape[1]-1:data.shape[1]].reshape(data.shape[0],1)
wall = []
ball = []
datas = [[],[],[],[],[],[],[],[],[],[]]
for i in range(0,data.shape[0]):
    label = int(data[i,data.shape[1]-1])
    datas[label].append(data[i,:])

for i in range(0,10):
    datas[i] = np.array(datas[i])

#print(datas[1][:,datas[i].shape[1]-1])


wall=[]
ball=[]
np.random.seed(7)
m = 200
c = 5000
alpw = 0.001
alpb = 0.001


for i in range(0,9):
    for j in range(i+1,10):
        
        datatemp = np.append(datas[i],datas[j],axis=0)
        np.random.shuffle(datatemp)
        x_train = datatemp[:,0:datatemp.shape[1]-1]
        y_train = datatemp[:,datatemp.shape[1]-1:datatemp.shape[1]]
        for z in range(0,x_train.shape[0]):
            if y_train[z,0]==i:
                y_train[z,0]=-1
            else:
                #print(y_train[z,0])
                y_train[z,0]=1
        #print(datatemp[:,784])
        w = np.random.randn(x_train.shape[1],1)  #784x1
        b = np.random.randn(1,1)  #4000x1
        for epoch in range(0,20):
            print(epoch)
            for k in range(0,int(x_train.shape[0]/m)):
                x = x_train[k*m:(k+1)*m,:]
                y = y_train[k*m:(k+1)*m,:]
                yhat = np.matmul(x,w)
                t = y*(yhat+b) #20000x1
                sgrt = np.zeros((x.shape[0],1))
                for j in range(0,x.shape[0]):
                    if t[j]>1:
                        sgrt[j]=0
                    else:
                        sgrt[j]=-1
                temp = sgrt*y
                gradw = w + (c/m)*np.sum((x*temp),axis=0).reshape(w.shape[0],1)
                #gradw = w + (c/m)*np.matmul(x_train.T,np.multiply(sgrt,y_train))
                #gradb = (c/m)*np.matmul(sgrt.T,y_train)
                gradb = (c/m)*np.sum(temp)
                w = w - alpw*gradw
                b = b - alpb*gradb
        wall.append(w)
        ball.append(b)


testdata = np.copy(train_data);

x_test = testdata[:,0:testdata.shape[1]-1]

y_pred = np.zeros((x_test.shape[0],len(wall)))
for j in range(0,len(wall)):
    y_pred[:,j]=(np.matmul(x_test,wall[j])+float(ball[j])).reshape(y_pred.shape[0])
count = 0
for a in range(0,9):
    for b in range(a+1,10):
        for m in range(0,y_pred.shape[0]):
            if y_pred[m,count]<0:
               y_pred[m,count]=a
            else:
                y_pred[m,count]=b
        count = count + 1
        print(count)

ans = []
for i in range(0,y_pred.shape[0]):
    counts = np.bincount(y_pred[i].astype(int))    
    ans.append(np.argmax(counts))\

        
from sklearn.metrics import accuracy_score

labels = np.array(labels)
y = np.array(labels)
print(accuracy_score(y, ans))
