# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 08:55:52 2017

@author: galad-loth
"""

import mxnet as mx

def SharedFeatureNet(data,conv_weight, conv_bias):
    shared_net = mx.sym.Convolution(data=data, kernel=(7, 7), stride=(1,1),
                         pad=(3, 3), num_filter=24,weight=conv_weight[0],
                        bias=conv_bias[0],name="conv0")
    shared_net = mx.sym.Activation(data=shared_net, act_type="relu", name="relu0")
    shared_net = mx.sym.Pooling(data=shared_net, kernel=(3,3),pool_type="max",                           
                           stride=(2,2), name="maxpool0")
    shared_net = mx.sym.Convolution(data=shared_net, kernel=(5, 5), stride=(1,1),
                         pad=(2, 2), num_filter=64, weight=conv_weight[1],
                        bias=conv_bias[1],name="conv1")
    shared_net = mx.sym.Activation(data=shared_net, act_type="relu", name="relu1")
    shared_net = mx.sym.Pooling(data=shared_net, kernel=(3,3),pool_type="max",                           
                           stride=(2,2), name="maxpool1")
    shared_net = mx.sym.Convolution(data=shared_net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=96, weight=conv_weight[2],
                        bias=conv_bias[2],name="conv2")
    shared_net = mx.sym.Activation(data=shared_net, act_type="relu", name="relu2")
    shared_net = mx.sym.Convolution(data=shared_net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=96, weight=conv_weight[3],
                        bias=conv_bias[3],name="conv3")
    shared_net = mx.sym.Activation(data=shared_net, act_type="relu", name="relu3")
    shared_net = mx.sym.Convolution(data=shared_net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[4],
                        bias=conv_bias[4],name="conv4")
    shared_net = mx.sym.Activation(data=shared_net, act_type="relu", name="relu4")
    shared_net = mx.sym.Pooling(data=shared_net, kernel=(3,3),pool_type="max",                           
                           stride=(2,2), name="maxpool4")    
    return shared_net

def MatchNetSymbol():
    data1=mx.sym.Variable("data1")
    data2=mx.sym.Variable("data2")
    dim_bottleneck=256
    conv_weight = []
    conv_bias = []
    for i in range(5):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))    
    conv_res1=SharedFeatureNet(data1,conv_weight,conv_bias)
    conv_res2=SharedFeatureNet(data2,conv_weight,conv_bias)
    
    botteneck_weights=mx.sym.Variable("botteneck_weights")
    botteneck_bias=mx.sym.Variable("botteneck_bias")
    feat1 = mx.sym.FullyConnected(data=conv_res1,num_hidden=dim_bottleneck, 
                                  weight=botteneck_weights,bias=botteneck_bias,
                                  name="botteneck")
    feat2 = mx.sym.FullyConnected(data=conv_res2,num_hidden=dim_bottleneck, 
                                  weight=botteneck_weights,bias=botteneck_bias,
                                  name="botteneck")
    
    conv_res=mx.sym.Concat(feat1,feat2,dim=1, name='conv_res')
    net = mx.sym.FullyConnected(data=conv_res,num_hidden=256, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc1_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=256, name="fc2")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc2_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=2, name="fc3")
    net = mx.symbol.Softmax(data=net,name="softmax")
    return net
    
if __name__=="__main__":
    matchnet=MatchNetSymbol()
    matchnet_ex=matchnet.simple_bind(ctx=mx.cpu(), data1=(50,1,64,64),
                                     data2=(50,1,64,64),softmax_label=(50,))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    