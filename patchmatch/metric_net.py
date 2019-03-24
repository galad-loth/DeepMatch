# -*- codingL utf-8-*-
"""
Created on Tue Sep 25 11:28:35 2018
@author: galad-loth
"""

import mxnet as mx
from feat_net import featnet1, featnet3,featnet5
import custops

def deep_compare_net_2ch():
    data = mx.sym.Variable("data")
    conv_weight=[]
    conv_bias=[]
    for i in range(3):
        conv_weight.append(mx.sym.Variable("conv"+str(i)+"_weight"))
        conv_bias.append(mx.sym.Variable("conv"+str(i)+"_bias"))
    net=featnet1(data, conv_weight, conv_bias, "feat_")
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=256, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc1_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=1, name="fc2")
    return net
 

def deep_compare_net_2ch_deep():
    data = mx.sym.Variable("data")
    conv_weight=[]
    conv_bias=[]
    for i in range(7):
        conv_weight.append(mx.sym.Variable("conv"+str(i)+"_weight"))
        conv_bias.append(mx.sym.Variable("conv"+str(i)+"_bias"))
    net=featnet3(data, conv_weight, conv_bias, "feat_")
    net = mx.sym.Flatten(data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=512, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc1_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=128, name="fc2")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc2_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=1, name="fc3")
    return net    

def deep_compare_net(net_type, is_train):
    if net_type == "2ch":
        net=deep_compare_net_2ch()
        if is_train:
            net=mx.symbol.Custom(data=net, name="loss", op_type="cus_hinge_loss")
        return net
    if net_type == "2ch_deep":
        net=deep_compare_net_2ch_deep()
        if is_train:
           net=mx.symbol.Custom(data=net, name="loss", op_type="cus_hinge_loss")
        return net

def match_net(feat_dim, fc_dim):
    data1 = mx.sym.Variable("data1")
    data2 = mx.sym.Variable("data2")
    conv_weight=[]
    conv_bias=[]
    for i in range(6):
        conv_weight.append(mx.sym.Variable("conv"+str(i)+"_weight"))
        conv_bias.append(mx.sym.Variable("conv"+str(i)+"_bias"))
    feat1=featnet5(data1, conv_weight, conv_bias,feat_dim, "feat1_") #two braches with share parameters
    feat2=featnet5(data2, conv_weight, conv_bias,feat_dim,"feat2_") #two braches with share parameters
    feat=mx.sym.concat(feat1,feat2,dim=1)
    net = mx.sym.FullyConnected(data=feat,num_hidden=fc_dim, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc1_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=fc_dim, name="fc2")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc2_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=2, name="fc3")
    net  = mx.sym.SoftmaxOutput(data=net, name='loss')
    return net
        
if __name__ == "__main__":
#    net=deep_compare_net("2ch", 1)
#   ex=net.simple_bind(ctx=mx.gpu(), data=(5, 3, 64, 64))
    net=match_net(512,256)
    ex=net.simple_bind(ctx=mx.gpu(),data1=(5, 1, 64, 64),data2=(5, 1, 64, 64))
    
    
    
    
    
    