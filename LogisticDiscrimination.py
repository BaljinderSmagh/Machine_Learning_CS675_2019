#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:37:46 2020

@author: baljindersmagh
"""

import sys
import numpy as np
import random
import math
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
       
        for j in range(len(self.traindata[0])):
            (self.w).append(random.uniform(-.01,.01))
            
    def Dotproduct(self,x,y):
        dp=0
        for i in range(len(y)):
            dp+=(x[i] * y[i])
        return dp
        
    def Sigmoid(self,x,y):
        return (1/(1+math.exp(-(self.Dotproduct(x,y))))) 
    
    '''def Sigmoid_2(self,x,y):
        return ((math.exp(-self.Dotproduct(x,y)))/(1+math.exp(-self.Dotproduct(x,y))))'''
         

    
    
    
    def Gradient(self):
        #dellf=(y-sigmoid) * x
        k=len(self.traindata[0])
        self.gradient=[0] * k
        for i in range(len(self.traindata)):
            sig=self.Sigmoid(self.w,self.traindata[i])
            for j in range(len(self.traindata[0])):
                #print(sig,'sig')
                self.gradient[j]+=((self.label[i][0]-sig)*self.traindata[i][j])
                
    def update_w(self):
        eta=0.001
        for i in range(len(self.traindata[0])):
            self.w[i]+= eta * self.gradient[i]  
            
       
            
    def Objective(self):
        #((-y * log (sigmoid))-(1-y) * log(1-sigmoid)))
        self.objective=0
        for i in range(len(self.traindata)):
            sig=self.Sigmoid(self.w,self.traindata[i])
            sig2=1-(sig)
            self.objective+=(-self.label[i][0] * math.log(sig) - (1- self.label[i][0]) * math.log(sig2))
         
              
    def train(self):
         self.dataDivider()
         prev_obj=float("inf")
         while True:
             self.Gradient()
             #print(self.gradient,'Gradient')
             self.update_w()
             self.Objective()
            
             print(self.objective,'Objective')
             #print(abs(prev_obj - self.objective),'diff')
             if (prev_obj - self.objective)<0.001:
                 #print('break')
                 #print(abs(prev_obj - self.objective),'diffbreak')
                 break
                 
             prev_obj=self.objective
             #print(prev_obj,'prev_obj')
         dist=0
         print(len(self.w),'len')
         print(self.w)
         for j in range(self.cols):
            dist += (self.w[j] ** 2)
         dist = np.sqrt(dist)
         print(dist,'||w||')
         print(self.w[-1],'w')
         print('dist: ',(self.w[-1])/dist)
         
    def predict(self):
        d=[0]
        for i in range(len(self.testdata)):
            d=np.dot(self.w,self.testdata[i])
            if d>0:
                self.predictlabels[i][0]=1
            else:
                self.predictlabels[i][0]=0
        
        print(self.predictlabels)

    def testing(self):
        count_1=0
        for i in range(len(self.predictlabels)):
            y=int(self.predictlabels[i][1])
            if self.predictlabels[i][0]==self.test[y][0]:
                count_1+=1
        Accuracy=(count_1/len(self.predictlabels)) * 100
        print('count',count_1)
        print(len(self.predictlabels),'len')
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
