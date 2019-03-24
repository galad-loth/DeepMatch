# -*- coding utf-8-*-
"""
Created on Tue Nov 23 10:15:35 2018
@author: galad-loth
"""

import numpy as npy
import mxnet as mx

class SSDHLoss(mx.operator.CustomOp):
    """
    Loss layer for supervised semantics-preserving deep hashing. 
    """
    def __init__(self, w_bin, w_balance):
        self._w_bin = w_bin
        self._w_balance = w_balance

    def forward(self, is_train, req, in_data, out_data, aux):
        x=in_data[0].asnumpy()
        xs=x-0.5
        y=npy.ones(x.shape)
        y[xs<0]=0
        self.assign(out_data[0], req[0], mx.nd.array(y))
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):        
        x=in_data[0].asnumpy()
        grad1=-2*(x-0.5)/x.shape[0]
        mu=npy.mean(x,axis=1)
        grad2=2*(mu-0.5)/x.shape[0]           
        grad=self._w_bin*grad1+self._w_balance*grad2[:,npy.newaxis] 
        self.assign(in_grad[0], req[0], mx.nd.array(grad))      

@mx.operator.register("ssdh_loss")
class SSDHLossProp(mx.operator.CustomOpProp):
    def __init__(self, w_bin, w_balance):
        super(SSDHLossProp, self).__init__(need_top_grad=False)
        self._w_bin=float(w_bin)
        self._w_balance=float(w_balance)
        
    def list_arguments(self):
        return ['data']
        
    def list_outputs(self):
        return ['output']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        return [data_shape],[data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return SSDHLoss(self._w_bin, self._w_balance)  
 
class SiamDHLoss(mx.operator.CustomOp):
    """
    Loss layer for deep hashing with siamese feature network. 
    """
    def __init__(self, margin, alpha):
        self._margin = margin
        self._alpha = alpha

    def forward(self, is_train, req, in_data, out_data, aux):
        x0=in_data[0].asnumpy()
        x1=in_data[1].asnumpy()
        l=in_data[0].asnumpy()
        d=self._alpha*(x0-x1)*(x0-x1)/2
        y=npy.zeros(x0.shape)
        y[l>0]=d[l>0]#same class, 
        y[l<=0]=npy.maximum(0, self.margin - d[l<=0])
        self.assign(out_data[0], req[0], mx.nd.array(y))
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):        
        x0=in_data[0].asnumpy()
        x1=in_data[1].asnumpy()        
        l=in_data[2].asnumpy()   
        y=out_data[0].asnumpy()        
       
        d=self._alpha*(x0-x1)
        dx0=npy.zeros(d.shape)
        dx1=npy.zeros(d.shape)
        
        dx0[l>0]=d[l>0]
        dx0[l<=0]=-d[l<=0]
        dx0[y<=0]=0.0
        dx1[:]=-dx0

        self.assign(in_grad[0], req[0], mx.nd.array(dx0)) 
        self.assign(in_grad[1], req[0], mx.nd.array(dx1))   

@mx.operator.register("siam_dh_Loss")
class SiamDHLossProp(mx.operator.CustomOpProp):
    def __init__(self, margin, alpha):
        super(SSDHLossProp, self).__init__(need_top_grad=False)
        self._margin = margin
        self._alpha = alpha
        
    def list_arguments(self):
        return ['data1','data2','label']
        
    def list_outputs(self):
        return ['output']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        label_shape=(in_shape[0][0],)
        return [data_shape, data_shape,label_shape],[data_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return SSDHLoss(self._margin, self._alpha)
        
        
        
def get_ssdh_symbol(net_pre,arg_params, 
                         num_latent, num_class,layer_name='flatten'):
    """
    net_pre: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_latent: the number of latent layer units for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = net_pre.get_internals()
    load_net = all_layers[layer_name+'_output']
    latent = mx.symbol.FullyConnected(data=load_net, num_hidden=num_latent, name='fc_ssdh_1')
    latent = mx.sym.Activation(data=latent, act_type="sigmoid", name="sigmoid_ssdh")
    class_net = mx.symbol.FullyConnected(data=latent, num_hidden=num_class, name='fc_ssdh_2')
    class_net = mx.symbol.SoftmaxOutput(data=class_net, name='softmax')
    
    hash_net=mx.symbol.Custom(data=latent, w_bin=0.1, w_balance =0.1,
                                  name="hash_loss", op_type="ssdh_loss")
    
    net = mx.sym.Group([class_net,hash_net])
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)        
        
        
if __name__=="__main__":
    data=mx.sym.Variable("data")
    loss=mx.symbol.Custom(data=data, w_bin=0.1, w_balance =0.1,
                                  name="loss", op_type="ssdh_loss")
    x=mx.nd.array([[0.5,0.3],[2,5],[-0.2,-3],[0.5,2]])
    print("x={}".format(x))
    x_grad=mx.nd.zeros((4,2))
    e= loss.bind(mx.cpu(), args={'data':x},args_grad={"data":x_grad})
    e.forward()
    print("out={}".format(e.outputs[0].asnumpy()))
    e.backward()
    print("x_grad={}".format(x_grad.asnumpy()))        
        
        
        
        
        
        
        
        
        
        
        
        
        