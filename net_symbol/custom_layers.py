# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 08:02:59 2017

@author: Fengjilan
"""
import numpy as npy
import mxnet as mx

class DeepCompareLossLayer(mx.operator.NumpyOp):
    def __init__(self):
        super(DeepCompareLossLayer, self).__init__(False)
        
    def list_arguments(self):
        return ['data','label']
        
    def list_outputs(self):
        return ['outputs']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        label_shape=(in_shape[0][0],)
        return [data_shape, label_shape],[data_shape]
        
    def forward(self, in_data, out_data):
        x=in_data[0]
#        l=in_data[1]
        y=out_data[0]
        y[:]=x
#        y[:]=npy.ones((x.shape[0],1))-l.reshape((x.shape[0],1))*x  
        
    def backward(self, out_grad, in_data, out_data, in_grad):
        x=in_data[0]
        l=in_data[1]
        y = npy.ones((x.shape[0],1))-l.reshape((x.shape[0],1))*x  
        dx=in_grad[0]
        
        dx[:]=1.0
        dx[l>0]=-1.0
        dx[y<0]=0.0