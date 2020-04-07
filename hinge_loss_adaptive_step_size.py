#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:26:00 2020

@author: baljindersmagh
"""



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
        
        #converting label of 0 to -1
        for k in range(len(self.label)):
            if self.label[k][0]==0:
               self.label[k][0]=-1
      
        x_0=np.ones((len(self.traindata),1))
        x_1=np.ones((len(self.testdata),1))
        #adding bias term
        print(self.testdata)
        print(self.traindata)
        self.testdata=np.hstack((self.testdata,x_1))
        self.traindata=np.hstack((self.traindata,x_0))
        print(self.traindata.shape)
        for j in range(len(self.traindata[0])):
            (self.w).append(random.uniform(-0.01,0.01))
    def Gradient(self):
        #dellf=(-xy)
        k=len(self.traindata[0])
        self.gradient=[0] * k
        for i in range(len(self.traindata)):
            for j in range(len(self.traindata[0])):
                if ((self.label[i][0]) * (np.dot(self.w,self.traindata[i])))<1:
                        self.gradient[j]-=(-(self.label[i][0] * self.traindata[i][j]))
                        
                        
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
            self.w[i]+= eta * self.gradient[i]  
            
       
            
    def Objective(self):
        #obj=(1-y(wx))
        self.objective=0    
        for i in range(len(self.traindata)):
            if ((self.label[i][0]) * (np.dot(self.w,self.traindata[i])))<1:
                self.objective+=(1-((np.dot(self.w,self.traindata[i])) * (self.label[i][0])))
           
              
    def train(self):
         self.dataDivider()
         prev_obj=float("inf")
         while True:
             self.Gradient()
             #print(self.gradient,'Gradient')
             
             self.Objective()
             self.update_w()
             print(self.objective,'Objective')
             print(abs(prev_obj - self.objective),'diff')
             if (abs(prev_obj - self.objective))<0.001:
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
        
        print('Accuracy',Accuracy)    
        
        

if __name__ =='__main__':
    #trainData=np.loadtxt('breast_cancer.data')
    #trainData=np.loadtxt('testLeastSquares.data')
    trainData=np.loadtxt('ionosphere.data')
    #trainData=np.loadtxt('testSVM.data')
    #trainLabels=np.loadtxt('breast_cancer.trainlabels.0')
    #trainLabels=np.loadtxt('testLeastSquares.trainlabels.0')
    trainLabels=np.loadtxt('ionosphere.trainlabels.0')
    #trainLabels=np.loadtxt('testSVM.trainlabels')
    trainLabels=trainLabels[trainLabels[:,1].argsort()]
    #testLabels=np.loadtxt('testleastsquarelabels.txt')
    testLabels=np.loadtxt('ionosphere.labels')
    #testLabels=np.loadtxt('testleastsquarelabels.txt')
    #testLabels=np.loadtxt('breast_cancer.labels')
    #object declaration
    model=LeastSquare(trainData,trainLabels,testLabels)
    
    
    model.train()
    
    prediction=model.predict()
    
    model.testing()