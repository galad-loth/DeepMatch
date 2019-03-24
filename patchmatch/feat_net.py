# -*- codingL utf-8-*-
"""
Created on Tue Sep 25 11:28:35 2018
@author: galad-loth
"""

import mxnet  as mx

def featnet1(data, conv_weight, conv_bias, name_prefix):
    """
    Convolutional feature network used for patch descriptor learning.
    For the case of weight shareing in Siamse/Triplet structure, pre-defined weights and bias
    variables are required.

    The feature net is designed to process "big" patches (e.g. 64 x 64)
    args:
        data: input data
        conv_weight: pre-defined weights for convolutional layers
        conv_bias: pre-defined bias for convolutional layers
        name_prefix: when multi-branch is needed, this is used to distinguish variables
                                in different branches
    returns:
        A feature net with 3 convolutional layer and 2 pooling layer.
    """

    net=mx.sym.Convolution(data=data, kernel=(7,7), stride=(3,3),
                             pad=(3,3),num_filter=96, weight=conv_weight[0],
                             bias=conv_bias[0], name=name_prefix+"conv0")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu0")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2),name=name_prefix+"maxpool0")
    net=mx.sym.Convolution(data=net, kernel=(5,5),stride=(1,1),
                            pad=(2,2), num_filter=192, weight=conv_weight[1],
                            bias=conv_bias[1], name=name_prefix+"conv1")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu1")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2),name=name_prefix+"maxpool1")
    net=mx.sym.Convolution(data=net, kernel=(3,3),stride=(1,1),
                            pad=(1,1), num_filter=256, weight=conv_weight[2],
                            bias=conv_bias[2], name=name_prefix+"conv2")           
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu2")   
    return net
    
def featnet2(data, conv_weight, conv_bias, name_prefix):
    """
    Convolutional feature network used for patch descriptor learning.
    For the case of weight shareing in Siamse/Triplet structure, pre-defined weights and bias
    variables are required.

    The feature net is designed to process "mid-size" patches (e.g. 32 x 32)
    args:
        data: input data
        conv_weight: pre-defined weights for convolutional layers
        conv_bias: pre-defined bias for convolutional layers
        name_prefix: when multi-branch is needed, this is used to distinguish variables 
                                in different branches
    returns:
        A feature net with 4 convolutional layer and 2 pooling layer.
    """    
    net=mx.sym.Convolution(data=data, kernel=(5,5), stride=(1,1),
                            pad=(2,2), num_filter=96, weight=conv_weight[0],
                            bias=conv_bias[0], name=name_prefix+"conv0")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu0")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2), name=name_prefix+"maxpool0")
    net=mx.sym.Convolution(data=net, kernel=(3,3), stride=(1,1),
                            pad=(1,1), num_filter=96, weight=conv_weight[1],
                            bias=conv_bias[1], name=name_prefix+"conv1")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu1")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2), name=name_prefix+"maxpool1")                       
    net=mx.sym.Convolution(data=net, kernel=(3,3), stride=(1,1),
                            pad=(1,1), num_filter=192, weight=conv_weight[2],
                            bias=conv_bias[2], name=name_prefix+"conv2")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu2") 
    net=mx.sym.Convolution(data=net, kernel=(3,3), stride=(1,1),
                            pad=(1,1), num_filter=192, weight=conv_weight[3],
                            bias=conv_bias[3], name=name_prefix+"conv3")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu3")
    return net
   

def featnet3(data, conv_weight, conv_bias, name_prefix):
    """
    Convolutional feature network used for patch descriptor learning.
    For the case of weight shareing in Siamse/Triplet structure, pre-defined weights and bias
    variables are required.
    This feature net desgined to process "large-size" patches(e.g. 64 x 64)
    args:
        data: input data
        conv_weight: pre-defined weights for covolutional layers
        conv_bias: pre-defined bias for covolutional layers
        name_prefix: when multi-branch is needed, this is used to 
                     distinguish variables in different branches
    returns:
        A feature net with 7 convolutional layer and 1 pooling layer. 
    """    
    net = mx.sym.Convolution(data=data, kernel=(5, 5), stride=(3,3),
                         pad=(2, 2), num_filter=64, weight=conv_weight[0],
                        bias=conv_bias[0],name=name_prefix+"conv0")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu0")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[1],
                        bias=conv_bias[1],name=name_prefix+"s1_conv0")    
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"s1_relu0")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[2],
                        bias=conv_bias[2],name=name_prefix+"s1_conv1")    
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"s1_relu1")    
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[3],
                        bias=conv_bias[3],name=name_prefix+"s1_conv2")    
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"s1_relu2")      
    net = mx.sym.Pooling(data=net, kernel=(2,2),pool_type="max",                           
                           stride=(2,2), name=name_prefix+"maxpool0") 
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[4],
                        bias=conv_bias[4],name=name_prefix+"s2_conv0")    
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"s2_relu0")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[5],
                        bias=conv_bias[5],name=name_prefix+"s2_conv1")    
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"s2_relu1")    
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                         pad=(1, 1), num_filter=64, weight=conv_weight[6],
                        bias=conv_bias[6],name=name_prefix+"s2_conv2")    
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"s2_relu2")
    return net      


def featnet4(data, conv_weight, conv_bias, name_prefix):
    """
    Convolutional feature network used for patch descriptor learning.
    For the case of weight sharing in Siamse/Triplet structure, pre-defined weights
    and bias variables are required.
    This structure is used by PN-Net/DeepCD.     
    args:
        data: input data
        conv_weight: pre-defined weights for covolutional layers
        conv_bias: pre-defined bias for covolutional layers
        name_prefix: when multi-branch is needed, this is used to 
                     distinguish variables in different branches
    returns:
        A feature net with 2 convolutional layer and 1 pooling layer. 
    """        
    net=mx.sym.Convolution(data=data, kernel=(7,7), stride=(1,1), num_filter=32, 
                           weight=conv_weight[0], bias=conv_bias[0], name=name_prefix+"conv0")
    net=mx.sym.Activation(data=net, act_type="tanh", name=name_prefix+"tanh0")
    net = mx.sym.Pooling(data=net, kernel=(2,2),pool_type="max",                           
                         stride=(2,2), name=name_prefix+"maxpool0") 
    net=mx.sym.Convolution(data=net, kernel=(6,6), stride=(1,1), num_filter=64, 
                           weight=conv_weight[1], bias=conv_bias[1], name=name_prefix+"conv1") 
    net=mx.sym.Activation(data=net, act_type="tanh", name=name_prefix+"tanh1")
    return net 
 
def featnet5(data, conv_weight, conv_bias, feat_dim=512, name_prefix=""):
    """
    Convolutional feature network used for patch descriptor learning.
    For the case of weight sharing in Siamse/Triplet structure, pre-defined weights
    and bias variables are required.
    This structure is used by match_net.     
    args:
        data: input data
        conv_weight: pre-defined weights for covolutional layers
        conv_bias: pre-defined bias for covolutional layers
        feat_dim:the dimensional of the extracted feature
        name_prefix: when multi-branch is needed, this is used to 
                     distinguish variables in different branches
    returns:
        A feature net with 5 convolutional layer and 3 pooling layer. 
    """      
    net=mx.sym.Convolution(data=data, kernel=(7,7), stride=(1,1),
                             pad=(3,3),num_filter=24, weight=conv_weight[0],
                             bias=conv_bias[0], name=name_prefix+"conv0")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu0")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2),name=name_prefix+"maxpool0")
    net=mx.sym.Convolution(data=net, kernel=(5,5),stride=(1,1),
                            pad=(2,2), num_filter=64, weight=conv_weight[1],
                            bias=conv_bias[1], name=name_prefix+"conv1")
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu1")
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2),name=name_prefix+"maxpool1")
    net=mx.sym.Convolution(data=net, kernel=(3,3),stride=(1,1),
                            pad=(1,1), num_filter=96, weight=conv_weight[2],
                            bias=conv_bias[2], name=name_prefix+"conv2")           
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu2")   
    net=mx.sym.Convolution(data=net, kernel=(3,3),stride=(1,1),
                            pad=(1,1), num_filter=96, weight=conv_weight[3],
                            bias=conv_bias[3], name=name_prefix+"conv3")           
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu3") 
    net=mx.sym.Convolution(data=net, kernel=(3,3),stride=(1,1),
                            pad=(1,1), num_filter=96, weight=conv_weight[4],
                            bias=conv_bias[4], name=name_prefix+"conv4")           
    net=mx.sym.Activation(data=net, act_type="relu", name=name_prefix+"relu4") 
    net=mx.sym.Pooling(data=net, kernel=(2,2), pool_type="max",
                            stride=(2,2),name=name_prefix+"maxpool2")
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=feat_dim, weight=conv_weight[5],
                            bias=conv_bias[5],name=name_prefix+"bottleneck")
    return net
    
    
if __name__=="__main__":
    data=mx.sym.Variable("data")
    conv_weight=[]
    conv_bias=[]
    for i in range(6):
        conv_weight.append(mx.sym.Variable("conv"+str(i)+"_weight"))
        conv_bias.append(mx.sym.Variable("conv"+str(i)+"_bias"))
    feat_net=featnet5(data, conv_weight, conv_bias,name_prefix="feat_")
    ex=feat_net.simple_bind(ctx=mx.cpu(), data=(50, 3, 64, 64))