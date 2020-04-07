#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:01:13 2020

@author: baljindersmagh
"""
import sys
import numpy as np

class NaiveBayes:
    
    def __init__(self,data,label,test):
        self.data=data
        self.label=label
        self.testLabels=test
        self.rows=len(self.data)
        self.cols=len(self.data[0])
        self.traindata=[]
        self.testdata=[]
        self.predictlabels=np.empty(shape=((self.rows-len(self.label)),2))
    def dataDivider(self):
        x=0
        for i in range(self.rows):
            if i in self.label[:,1]:
                self.traindata.append(self.data[i])
            else:
                self.testdata.append(self.data[i])
                self.predictlabels[x]=[0,i]
                x+=1
              
        
        
    def meanCalculation(self):
        N=len(self.traindata[0])
        self.mean_0=[0.01]* N
        self.mean_1=[0.01]* N
        count_0=0
        count_1=0
        for i in range(len(self.traindata)):
            if self.label[i,0]==0:
                count_0+=1
                for j in range(len(self.traindata[0])):
                    self.mean_0[j]+=self.traindata[i][j]
            else:
                count_1+=1
                for j in range(len(self.traindata[0])):
                    self.mean_1[j]+=self.traindata[i][j]
        for i in range(len(self.mean_0)):
            self.mean_0[i]=(self.mean_0[i])/count_0 
            self.mean_1[i]=(self.mean_1[i])/count_1 
    def varianceCalculation(self):
        N=len(self.traindata[0])
        self.variance_0=[0] * N
        self.variance_1=[0] * N
        count_0=0
        count_1=0
        for i in range(len(self.traindata)):
            if self.label[i,0]==0:
                count_0+=1
                for j in range(len(self.traindata[0])):
                    self.variance_0[j]+=((self.traindata[i][j]-self.mean_0[j])**2)
            else:
                count_1+=1
                for j in range(len(self.traindata[0])):
                    self.variance_1[j]+=((self.traindata[i][j]-self.mean_1[j])**2)
                    
        for i in range(len(self.variance_0)):           
            self.variance_0[i]=(self.variance_0[i])/count_0 
            self.variance_1[i]=(self.variance_1[i])/count_1 
    
    def distance(self):
        M=len(self.testdata)
        self.dist_0=[0]* M
        self.dist_1=[0]* M
        #print(len(self.dist_0))
        for i in range(len(self.testdata)):
            for j in range(len(self.testdata[0])):
                self.dist_0[i]+=(((self.testdata[i][j]-self.mean_0[j]))**2/self.variance_0[j])

                self.dist_1[i]+=(((self.testdata[i][j]-self.mean_1[j]))**2/self.variance_1[j])
    
    def train(self):
        self.dataDivider()
        self.meanCalculation()
        self.varianceCalculation()
    def predict(self):
        self.distance()
        for i in range(len(self.testdata)):
            if self.dist_0[i]>self.dist_1[i]:
                self.predictlabels[i][0]=1
            else:
                self.predictlabels[i][0]=0
        
        print(self.predictlabels)    
       
    def testing(self,predict,labels):
        count_1=0
        Accuracy=0
        for i in range(len(self.predictlabels)):
            y=int(self.predictlabels[i][1])
            if self.predictlabels[i][0]==self.testLabels[y][0]:
                count_1+=1
        Accuracy=(count_1/len(self.predictlabels)) * 100
        print('Accuracy',Accuracy)    
        

if __name__ =='__main__':
    
    
    trainData=np.loadtxt(sys.argv[1])
   

    trainLabels=np.loadtxt(sys.argv[2])
    
    trainLabels=trainLabels[trainLabels[:,1].argsort()]
    testLabels=np.loadtxt(sys.argv[3])
    #object declaration
    model=NaiveBayes(trainData,trainLabels,testLabels)
    
    
    
    
    model.train()
    
    prediction=model.predict()
    
    model.testing(prediction,testLabels)
