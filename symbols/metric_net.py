# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 09:35:33 2017

@author: galad-loth
"""
import mxnet as mx
import feat_net

def metric_net_2ch():
    data = mx.sym.Variable("data")
    conv_weight = []
    conv_bias = []
    for i in range(3):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    conv_res= feat_net.featnet1(data,conv_weight, conv_bias,"")
    conv_res = mx.sym.Flatten(data=conv_res)
    net = mx.sym.FullyConnected(data=conv_res,num_hidden=256, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="relu3")
    net = mx.sym.FullyConnected(data=net,num_hidden=1, name="fc2")
    return net  


def metric_net_2ch_cs():
    datas= mx.sym.Variable("datas")
    datac= mx.sym.Variable("datac")
    conv_weight_s = []
    conv_bias_s = []
    conv_weight_c = []
    conv_bias_c = []
    for i in range(4):
        conv_weight_s.append(mx.sym.Variable('conv' + str(i) + '_weight_s'))
        conv_bias_s.append(mx.sym.Variable('conv' + str(i) + '_bias_s'))
        conv_weight_c.append(mx.sym.Variable('conv' + str(i) + '_weight_c'))
        conv_bias_c.append(mx.sym.Variable('conv' + str(i) + '_bias_c'))
    conv_res_s=feat_net.featnet2(datas,conv_weight_s, conv_bias_s,"bs_")
    conv_res_c=feat_net.featnet2(datac,conv_weight_c, conv_bias_c,"bc_")
    conv_res=mx.sym.Concat(conv_res_s,conv_res_c,dim=1, name='conv_res')
    net = mx.sym.FullyConnected(data=conv_res,num_hidden=768, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="relu1")
    net = mx.sym.FullyConnected(data=net,num_hidden=1, name="fc2")
    return net
    
if __name__=="__main__":
    net=metric_net_2ch_cs()
    ex=net.simple_bind(ctx=mx.cpu(), datas=(50,2,64,64),datac=(50,2,64,64))    
    