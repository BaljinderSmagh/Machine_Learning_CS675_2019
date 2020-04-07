#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:42:16 2020

@author: baljindersmagh
"""

import sys
import numpy as np
import random

class LeastSquare:
    
    def __init__(self,data,label,test):
        self.data=data
        self.label=label
        self.test=test
        self.rows=len(self.data)
        self.cols=len(self.data[0])
        self.traindata=[]
        self.testdata=[]
        self.w=[]
        self.predictlabels=np.empty(shape=((self.rows-len(self.label)),2))
        
        
    def dataDivider(self):
        x=0
        for i in range(self.rows):
            if i in self.label[:,1]:
                self.traindata.append(list(self.data[i]))
            else:
                self.testdata.append(list(self.data[i]))
                self.predictlabels[x]=[0,i]
                x+=1
                
        x_0=np.ones((len(self.traindata),1))
        x_1=np.ones((len(self.testdata),1))
        #adding bias term
        self.testdata=np.hstack((self.testdata,x_1))
        self.traindata=np.hstack((self.traindata,x_0))
        print(self.traindata.shape)
        for j in range(len(self.traindata[0])):
            (self.w).append(random.uniform(-0.01,0.01))
        #print(len(self.w))
    def Gradient(self):
        #dellf=(y-(wx))x
        k=len(self.traindata[0])
        self.gradient=[0] * k
        for i in range(len(self.traindata)):
            
            for j in range(len(self.traindata[0])):
                self.gradient[j]+=((self.label[i][0]-np.dot(self.w,self.traindata[i])) * self.traindata[i][j])
                
        
    
              
        
            
    def eta_selection(self):
        self.best_eta=0
        min_obj=1000000000000
        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
        for k in range(0, len(eta_list), 1):
            eta = eta_list[k]
            for i in range((len(self.traindata[0]))):
                self.w[i]+= eta * self.gradient[i]

            self.Objective()
            if (self.objective<min_obj):
                min_obj=self.objective
                best_eta=eta
            for i in range((len(self.traindata[0]))):
        
                self.w[i]-= eta * self.gradient[i]
        print('best_eta',best_eta)
        return best_eta
        
           
    
    
    def update_w(self):
        eta=self.eta_selection()
        for i in range(len(self.traindata[0])):
            self.w[i]+= ( self.gradient[i] * eta)
        return self.w
            
       
            
    def Objective(self):
        #obj=(y-(w+W_0)x)^2
        self.objective=0    
        for i in range(len(self.traindata)):
            self.objective+=(self.label[i][0]-np.dot(self.w,self.traindata[i]))**2
        return self.objective
    
    
    
    
    
    def train(self):
         self.dataDivider()
         prev_obj=float("inf")
         while True:
             self.Gradient()
             self.update_w()
             self.Objective()
             print(self.objective,'Objective')
             if (prev_obj-self.objective)<0.001:
                 break
             prev_obj=self.objective
         dist=0
         for j in range(self.cols):
            dist += (self.w[j] ** 2)
         dist = np.sqrt(dist)

         print('w: ',abs((self.w[-1])/dist))
         
    def predict(self):
        d=[0]
        for i in range(len(self.testdata)):
            d=np.dot(self.w,self.testdata[i])
            if d>0.5:
                self.predictlabels[i][0]=1
            else:
                self.predictlabels[i][0]=0
        
        print(self.predictlabels)

    def testing(self):
        count_1=0
        for i in range(len(self.predictlabels)):
            #y=int(self.predictlabels[i][1])
            #print('y',y)
            if self.predictlabels[i][0]==self.test[i][0]:
                count_1+=1
     
        Accuracy=(count_1/len(self.predictlabels)) * 100
        print('Accuracy',Accuracy)    
        
        

if __name__ =='__main__':
    
    trainData=np.loadtxt(sys.argv[1])
    
    trainLabels=np.loadtxt(sys.argv[2])
    trainLabels=trainLabels[trainLabels[:,1].argsort()]
    
    testLabels=np.loadtxt(sys.argv[3])
   
    #object declaration
    model=LeastSquare(trainData,trainLabels,testLabels)
    
    
    model.train()
    
    prediction=model.predict()
    
    model.testing()
