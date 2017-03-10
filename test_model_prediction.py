# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 21:00:11 2017

@author: galad-loth
"""

import numpy as npy
import mxnet as mx
import cv2
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

def predict(img, mod):
    img = cv2.resize(img, (224, 224))
    img = npy.swapaxes(img, 0, 2)
    img = npy.swapaxes(img, 1, 2) 
    img = img[npy.newaxis, :] 
    
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = npy.squeeze(prob)

    a = npy.argsort(prob)[::-1]    
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], i))
 

if __name__=="__main__":  
    net, arg_params, aux_params = mx.model.load_checkpoint('pretrain_model\\Inception-BN', 126)
    mod = mx.mod.Module(symbol=net, context=mx.gpu())
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    mod.set_params(arg_params, aux_params)   
    img=cv2.imread("data\\2008_001194.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict(img,mod)