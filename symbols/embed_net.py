# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 09:35:33 2017

@author: galad-loth
"""
import mxnet as mx
import feat_net

def embed_net_siam1():
    """
    This is a siamese type network to compare two patches.
    """
    data1 = mx.sym.Variable("data1")
    data2 = mx.sym.Variable("data2")
    conv_weight = []
    conv_bias = []
    for i in range(3):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    conv_res1= feat_net.featnet1(data1,conv_weight, conv_bias,"")   
    conv_res2= feat_net.featnet1(data2,conv_weight, conv_bias,"")
    conv_res=mx.sym.Concat(conv_res1,conv_res2,dim=1, name='conv_res')
    net = mx.sym.FullyConnected(data=conv_res,num_hidden=512, name="fc1")
    net = mx.sym.Activation(data=net, act_type="relu", name="fc1_relu")
    net = mx.sym.FullyConnected(data=net,num_hidden=1, name="fc2")
    return net


def embed_net_siam2():  
    """
    This network is similar to the one used in MatchNet.
    """
    data1 = mx.sym.Variable("data1")
    data2 = mx.sym.Variable("data2")
    conv_weight = []
    conv_bias = []
    for i in range(3):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    conv_res1= feat_net.featnet1(data1,conv_weight, conv_bias,"")   
    conv_res2= feat_net.featnet1(data2,conv_weight, conv_bias,"")
    botteneck_weights=mx.sym.Variable("botteneck_weights")
    botteneck_bias=mx.sym.Variable("botteneck_bias")
    feat1 = mx.sym.FullyConnected(data=conv_res1,num_hidden=256, 
                                  weight=botteneck_weights,bias=botteneck_bias,
                                  name="botteneck")
    feat2 = mx.sym.FullyConnected(data=conv_res2,num_hidden=256, 
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


   
#def embed_net_ssdh(net_pre,arg_params,num_latent, 
#                   num_class,layer_name='flatten'):
#    """
#    Net work of semantic-preserving deep hashing.
#    Args:
#        net_pre: the pre-trained network symbol
#        arg_params: the argument parameters of the pre-trained model
#        num_latent: the number of latent layer units for the fine-tune datasets
#        layer_name: the layer name before the last fully-connected layer
#    return:
#        
#    """
#    all_layers = net_pre.get_internals()
#    load_net = all_layers[layer_name+'_output']
#    latent = mx.symbol.FullyConnected(data=load_net, num_hidden=num_latent, name='latent_ssdh')
#    latent = mx.sym.Activation(data=latent, act_type="sigmoid", name="sigmoid_ssdh")
#    class_net = mx.symbol.FullyConnected(data=latent, num_hidden=num_class, name='fc_ssdh')
#    class_net = mx.symbol.SoftmaxOutput(data=class_net, name='softmax')
#    hash_loss=HashLossLayer(0.1,0.1)
#    hash_net=hash_loss(data=latent, name="hash")
#    net = mx.sym.Group([class_net,hash_net])
#    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
#    return (net, new_args) 

    
    
    
if __name__=="__main__":
    net=embed_net_siam2()
    ex=net.simple_bind(ctx=mx.cpu(), data1=(50,2,64,64), data2=(50,2,64,64))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    