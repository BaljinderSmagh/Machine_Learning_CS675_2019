#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:42:29 2020

@author: baljindersmagh
"""
import sys
import numpy as np
import math

class KNN:
    
    def __init__(self,data,label,test,k):
        self.data=data
        self.label=label
        self.test=test
        self.k=k
        self.rows=len(self.data)
        self.cols=len(self.data[0])
        self.traindata=[]
        self.testdata=[]
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
    
    
    
    def EuclideanDistance(self):
        D=len(self.traindata)
        for n in range(len(self.testdata)):
            #print('\n ******************************** \n')
            self.dist={}
            self.m=[0] * D
            for h in range(len(self.traindata)):
                for j in range(self.cols):
                    self.m[h]+=(((self.testdata[n][j]-self.traindata[h][j]))**2)
                self.m[h]=math.sqrt(self.m[h])
                #print('Dist:{} and {} ={}  give label:{}'.format(self.testdata[n],self.traindata[h],self.m[h],self.label[h][0]))
                self.dist[self.m[h]]=self.label[h][0]
            dist_list=sorted(self.dist.items(),reverse=False)
            count_0=0
            count_1=0
            for g in range(self.k):
                if dist_list[g][1]==0:
                    count_0+=1
                else:
                    count_1+=1
            if count_0>count_1:
                self.predictlabels[n][0]=0
            else:
                self.predictlabels[n][0]=1
        print(self.predictlabels)
            
    
    def train(self):
        self.dataDivider()
        self.EuclideanDistance()
        
    def predict(self):
        #print(self.predictlabels) 
        pass
       
    def testing(self):
        count_1=0
        Accuracy=0
        for i in range(len(self.predictlabels)):
            y=int(self.predictlabels[i][1])
            if self.predictlabels[i][0]==self.test[y][0]:
                count_1+=1
        Accuracy=(count_1/len(self.predictlabels)) * 100
        print('Accuracy',Accuracy)    
    

if __name__ =='__main__':
    trainData=np.loadtxt(sys.argv[1])
    

    trainLabels=np.loadtxt(sys.argv[2])
    
    
    trainLabels=trainLabels[trainLabels[:,1].argsort()]
    
    testLabels=np.loadtxt(sys.argv[3])
    #object declaration
    model=KNN(trainData,trainLabels,testLabels,k=20 )
    
    
    model.train()
    
    prediction=model.predict()
    
    model.testing()
