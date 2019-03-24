# -*- coding utf-8-*-
"""
Created on Tue Nov 23 10:15:35 2018
@author: galad-loth
"""

import mxnet as mx

def cifar10_featnet1(data, conv_weight, conv_bias, num_filter, dim_out, name_prefix):
    """
    Convolutional feature network used for binary code learning.

    The feature net is designed to process "small" patches (e.g. 32 x 32 for CIFAR-10)
    args:
      
    returns:
        A feature net with 4 convolutional layer and 2 pooling layer.
    """   
    net=mx.sym.Convolution(data=data, kernel=(5,5), stride=(1,1),
                            pad=(2,2), num_filter=num_filter//2, 
                            weight=conv_weight[0], bias=conv_bias[0],
                            name=name_prefix + "conv0")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix + "relu0")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                       stride=(2,2), name=name_prefix + "maxpool0")
    net=mx.sym.Convolution(data=net, kernel=(3,3), stride=(1,1),
                            pad=(1,1), num_filter=num_filter//2,
                            weight=conv_weight[1], bias=conv_bias[1],
                            name=name_prefix + "conv1")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix + "relu1")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="avg",
                       stride=(2,2), name=name_prefix + "maxpool1")                       
    net=mx.sym.Convolution(data=net, kernel=(3,3), stride=(1,1),
                            pad=(1,1), num_filter=num_filter, 
                            weight=conv_weight[2], bias=conv_bias[2],
                            name=name_prefix + "conv2")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix + "relu2") 
    net=mx.sym.Convolution(data=net, kernel=(3,3), stride=(1,1),
                            pad=(1,1), num_filter=num_filter,
                            weight=conv_weight[3], bias=conv_bias[3],
                            name=name_prefix + "conv3")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix + "relu3")
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=dim_out, name=name_prefix + "fc1")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix + "relu4")    
    return net

def residual_unit(data, num_filter, dim_match, name_prefix):
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), 
                               stride=(1,1), pad=(1,1),no_bias=True,
                                name=name_prefix + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=0.9, 
                           eps=2e-5, name=name_prefix + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name_prefix + '_relu1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3),
                               stride=(1,1), pad=(1,1),no_bias=True,
                               name=name_prefix + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=0.9, 
                           eps=2e-5, name=name_prefix + '_bn2')
    if dim_match:
        shortcut=data
    else:
        conv1sc = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1),
                                     stride=(1,1), no_bias=True,
                                     name=name_prefix+'_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, 
                                    momentum=0.9, eps=2e-5, name=name_prefix + '_bnsc')
    net=shortcut + bn2
    net=mx.sym.Activation(data=net, act_type='relu', name=name_prefix + '_relu')
    return net
    
        
def cifar10_featnet2(num_filter, dim_out):
    """
    Convolutional feature network used for binary code learning.

    The feature net is designed to process "small" patches (e.g. 32 x 32 for CIFAR-10)
    args:
      
    returns:
        A feature net with 4 convolutional layer and 2 pooling layer.
    """
    data=mx.sym.Variable('data')    
    net=mx.sym.Convolution(data=data, kernel=(5,5), stride=(1,1),
                            pad=(2,2), num_filter=num_filter//2, name="conv0")
    net=mx.sym.Activation(data=net, act_type="relu", name="relu0")
    net=mx.sym.Pooling(data=net, kernel=(3,3), pool_type="max",
                       stride=(2,2), name="maxpool0")
    net=residual_unit(data=net, num_filter=num_filter, dim_match=False, name_prefix="res1")
    net=residual_unit(data=net, num_filter=num_filter, dim_match=True, name_prefix="res2")
    net = mx.sym.Pooling(data=net, kernel=(3, 3),pool_type='avg', 
                         stride=(2,2), name='avgpool1')
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=dim_out, name="fc1")
    net=mx.sym.Activation(data=net, act_type="relu", name="relu4")   
    return net    
    
if __name__ == "__main__":
    data=mx.sym.Variable("data")
    conv_weight=[]
    conv_bias=[]
    for i in range(4):
        conv_weight.append(mx.sym.Variable("conv"+str(i)+"_weight"))
        conv_bias.append(mx.sym.Variable("conv"+str(i)+"_bias"))
    feat_net=cifar10_featnet1(data, conv_weight, conv_bias, 64, 512, "feat_")
    ex=feat_net.simple_bind(ctx=mx.cpu(), data=(50, 3, 64, 64))
#    net=cifar10_featnet2(64, 512)
#    ex=net.simple_bind(ctx=mx.cpu(), data=(50, 3, 64, 64))