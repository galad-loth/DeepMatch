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
        
class TripletLossLayer(mx.operator.NumpyOp):
    def __init__(self, margin):
        super(TripletLossLayer, self).__init__(False)
        self.margin=margin
        
    def list_arguments(self):
        return ['data0','data1','data2']
        
    def list_outputs(self):
        return ['triplet_loss']
        
    def infer_shape(self, in_shape):
        data_shape0=in_shape[0]
        data_shape1=in_shape[1]
        data_shape2=in_shape[2]
        if (data_shape0!=data_shape1) or (data_shape0!=data_shape2):
            raise ValueError("Shape of inputs does not match:"
                              "{}{}{}".format(data_shape0, data_shape1,data_shape2))            
        
        return [data_shape0, data_shape1,data_shape2],[data_shape0[0]]
        
    def forward(self, in_data, out_data):
        x0=in_data[0]
        x1=in_data[1]
        x2=in_data[2]
        y=out_data[0]
        d1=x0-x1
        d2=x0-x2
        d1=d1*d1
        d2=d2*d2
        loss=d1-d2+self.margin
        y[:]=loss
        y[loss<0]=0
#        y[:]=npy.ones((x.shape[0],1))-l.reshape((x.shape[0],1))*x  
        
    def backward(self, out_grad, in_data, out_data, in_grad):
        x0=in_data[0]
        x1=in_data[1]
        x2=in_data[2]
        y=out_data[0]
        dx0=in_grad[0]
        dx1=in_grad[1]
        dx2=in_grad[2]
        dx0[:]=2*(x2-x1)
        dx1[:]=2*(x1-x0)
        dx2[:]=2*(x0-x2)
        dx0[y<=0]=0
        dx1[y<=0]=0
        dx2[y<=0]=0

class TripletRatioLossLayer(mx.operator.NumpyOp):
    def __init__(self, margin):
        super(TripletRatioLossLayer, self).__init__(False)
        self.margin=margin
        
    def list_arguments(self):
        return ['data0','data1','data2']
        
    def list_outputs(self):
        return ['triplet_ratio_loss']
        
    def infer_shape(self, in_shape):
        data_shape0=in_shape[0]
        data_shape1=in_shape[1]
        data_shape2=in_shape[2]
        if (data_shape0!=data_shape1) or (data_shape0!=data_shape2):
            raise ValueError("Shape of inputs does not match:"
                              "{}{}{}".format(data_shape0, data_shape1,data_shape2))            
        
        return [data_shape0, data_shape1,data_shape2],[data_shape0[0]]
        
    def forward(self, in_data, out_data):
        x0=in_data[0]
        x1=in_data[1]
        x2=in_data[2]
        y=out_data[0]
        dp=x0-x1
        dn=x0-x2
        dp=dp*dp
        dn=dn*dn
        loss=npy.maximum(0, 1-dn/(dp+self.margin))
        y[:]=loss
        
#        y[:]=npy.ones((x.shape[0],1))-l.reshape((x.shape[0],1))*x  
        
    def backward(self, out_grad, in_data, out_data, in_grad):
        x0=in_data[0]
        x1=in_data[1]
        x2=in_data[2]
        y=out_data[0]
        dx0=in_grad[0]
        dx1=in_grad[1]
        dx2=in_grad[2]
        dp=x0-x1
        dn=x0-x2
        temp=dp*dp+self.margin
        
        dx1[:]=-2*dn*dn*dp/temp/temp
        dx2[:]=2*dn/temp
        dx0[:]=-dx1-dx2
        dx0[y<=0]=0
        dx1[y<=0]=0
        dx2[y<=0]=0
        


     




















