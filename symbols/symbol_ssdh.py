# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 21:00:11 2017

@author: galad-loth
"""
import numpy as npy
import mxnet as mx

class HashLossLayer(mx.operator.NumpyOp):
    def __init__(self, w_bin,w_balance):
        super(HashLossLayer, self).__init__(False)
        self.w_bin=w_bin
        self.w_balance=w_balance
        
    def list_arguments(self):
        return ['data']
        
    def list_outputs(self):
        return ['output']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        return [data_shape],[data_shape]
        
    def forward(self, in_data, out_data):
        x=in_data[0]
#        l=in_data[1]
        y=out_data[0]
        xs=x-0.5
        y[:]=1
        y[xs<0]=0
#        y[:]=npy.ones((x.shape[0],1))-l.reshape((x.shape[0],1))*x  
        
    def backward(self, out_grad, in_data, out_data, in_grad):
        x=in_data[0]
        dx=in_grad[0]
        
        grad1=-2*(x-0.5)/x.shape[1]
        mu=npy.mean(x,axis=1)
        grad2=2*(mu-0.5)/x.shape[1]
        
        grad=self.w_bin*grad1+self.w_balance*grad2
        dx[:]=grad


def get_finetune_symbol(net_pre,arg_params, 
                         num_latent, num_class,layer_name='flatten'):
    """
    net_pre: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_latent: the number of latent layer units for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = net_pre.get_internals()
    load_net = all_layers[layer_name+'_output']
    latent = mx.symbol.FullyConnected(data=load_net, num_hidden=num_latent, name='latent_ssdh')
    latent = mx.sym.Activation(data=latent, act_type="sigmoid", name="sigmoid_ssdh")
    class_net = mx.symbol.FullyConnected(data=latent, num_hidden=num_class, name='fc_ssdh')
    class_net = mx.symbol.SoftmaxOutput(data=class_net, name='softmax')
    hash_loss=HashLossLayer(0.1,0.1)
    hash_net=hash_loss(data=latent, name="hash")
    net = mx.sym.Group([class_net,hash_net])
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)
    