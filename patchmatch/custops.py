# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:28:35 2018
@author: galad-loth
"""

import numpy as npy
import mxnet as mx
   
class CusSiameseEmbedLoss(mx.operator.CustomOp):
    """
    L2 loss layer for patch descriptor learning based on Siamese type network.
    Two patches are feed into two branch of convolutional network with shared weights.
    This loss tries to minimize the difference between the two feature vectors extraced by 
    the Siamese network. 
    """
    
    def forward(self, is_train, req, in_data, out_data, aux):
        x0=in_data[0].asnumpy()
        x1=in_data[1].asnumpy()
        l=in_data[2].asnumpy()
        
        d=(x0-x1)*(x0-x1)
        y=npy.zeros(x0.shape)
        y[l>0]=d[l>0]
        y[l<=0]=npy.maximum(0, self.margin - d[l<=0])

        self.assign(out_data[0], req[0], mx.nd.array(y))
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):        
        x0=in_data[0].asnumpy()
        x1=in_data[1].asnumpy()
        l=in_data[2] .asnumpy()   
        y=out_data[0].asnumpy()        
       
        d=2*(x0-x1)
        dx0=npy.zeros(d.shape)
        dx1=npy.zeros(d.shape)
        
        dx0[l>0]=d[l>0]
        dx0[l<=0]=-d[l<=0]
        dx0[y<= 0]=0.0
        dx1[:]=-dx0

        self.assign(in_grad[0], req[0], mx.nd.array(dx0))
        self.assign(in_grad[1], req[0], mx.nd.array(dx1))       

@mx.operator.register("cus_siam_loss")
class CusSiameseEmbedLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CusSiameseEmbedLossProp, self).__init__(need_top_grad=False)
        
    def list_arguments(self):
        return ['data0','data1', 'label']
        
    def list_outputs(self):
        return ['outputs']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        label_shape=(in_shape[0][0],)
        return [data_shape, data_shape, label_shape],[data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CusSiameseEmbedLoss()  
    
    
class CusTripletLoss(mx.operator.NumpyOp):
    """
    Triplet loss layer for patch descriptor learning.
    This loss tries to ensure that patches of the same 3D point is more similar 
    than patches of different 3D points. 
    """
    def forward(self, is_train, req, in_data, out_data, aux):
        x0=in_data[0].asnumpy()
        x1=in_data[1].asnumpy()
        x2=in_data[2].asnumpy()
        d1=x0-x1
        d2=x0-x2
        d1=d1*d1
        d2=d2*d2
        y=d1-d2+self.margin
        y[y<0]=0
        self.assign(out_data[0], req[0], mx.nd.array(y))
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):     
        x0=in_data[0].asnumpy()
        x1=in_data[1].asnumpy()
        x2=in_data[2].asnumpy()
        y=out_data[0].asnumpy()
#        dx0=in_grad[0]
#        dx1=in_grad[1]
#        dx2=in_grad[2]
        dx0=2*(x2-x1)
        dx1=2*(x1-x0)
        dx2=2*(x0-x2)
        dx0[y<=0]=0
        dx1[y<=0]=0
        dx2[y<=0]=0 
        self.assign(in_grad[0], req[0], mx.nd.array(dx0))
        self.assign(in_grad[1], req[0], mx.nd.array(dx1)) 
        self.assign(in_grad[2], req[0], mx.nd.array(dx2)) 
    
@mx.operator.register("cus_triplet_loss")
class CusTripletLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CusTripletLossProp, self).__init__(need_top_grad=False)
        
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

    def create_operator(self, ctx, shapes, dtypes):
        return CusTripletLoss()  


class CusHingeLoss(mx.operator.CustomOp):
    """
    Hinge-base loss layer for patch descriptor learning.
    This loss is used in the method DeepCompare, in which the network predict the 
    similarity of two patches based on the feature extracted with convolutional layers.  
    """
    
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x=in_data[0].asnumpy()
        l=in_data[1].asnumpy()
        
        y=npy.ones(x.shape)-l.reshape((x.shape[0],1))*x
        dx=npy.ones(x.shape)
        dx[l>0]=-1.0
        dx[y<0]=0.0
        self.assign(in_grad[0], req[0], mx.nd.array(dx))

@mx.operator.register("cus_hinge_loss")
class CusHingeLossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CusHingeLossProp, self).__init__(need_top_grad=False)
        
    def list_arguments(self):
        return ['data','label']
        
    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        label_shape=(in_shape[0][0],)
        return [data_shape, label_shape],[data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CusHingeLoss()  
    
if __name__=="__main__":
    d=mx.symbol.Variable("data")
    loss=mx.symbol.Custom(data=d, op_type="cus_hinge_loss", name="loss")
    x=mx.nd.array([[0.5],[5],[-2],[2]])
    l=mx.nd.array([1,1,-1,-1])
    x_grad=mx.nd.zeros((4,1))
    e= loss.bind(mx.cpu(), args={'data':x, 'loss_label':l},args_grad={"data":x_grad})
    e.forward()
    print("out={}".format(e.outputs[0].asnumpy()))
    e.backward()
    print("x_grad={}".format(x_grad.asnumpy()))
    

   
    
    
    
    
    
    
    
    
    
    
    
    