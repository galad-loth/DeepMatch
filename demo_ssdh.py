# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 20:26:17 2017

@author: galad-loth
"""

import numpy as npy
import mxnet as mx
import logging
import sys
from net_symbol.symbol_ssdh import get_finetune_symbol
from utils.data import GetImgClassIter
from utils.evaluate_metric import MyAccuracy

#logging.basicConfig(level=logging.INFO)

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

load_net, load_arg_params, load_aux_params = \
    mx.model.load_checkpoint('pretrain_model\\Inception-BN', 126)
new_net, load_args=get_finetune_symbol(load_net,load_arg_params,512,45)

batch_size=20
datadir="E:\\DevProj\\Datasets\\NWPU-RESISC45\\images"
trainIter,valIter, cls_dict=GetImgClassIter(datadir,(batch_size,3,256,256),True,0.2)

model = mx.mod.Module(symbol= new_net, context= mx.cpu())

#optimizer_params = {
#            'learning_rate': 0.01,
#            'momentum' : 0.9,
#            'wd' : 0.005,
#            'lr_scheduler': mx.lr_scheduler.FactorScheduler(5000,0.9)}
optimizer = mx.optimizer.create('sgd',
                                learning_rate =0.001,
                                momentum = 0.9,
                                wd = 0.001,
                                lr_scheduler=mx.lr_scheduler.FactorScheduler(1000,0.9))
new_net_args=new_net.list_arguments()
lr_scale={}
for arg_name in new_net_args:
    if "ssdh" in arg_name:
        lr_scale[arg_name] = 10
optimizer.set_lr_mult(lr_scale)
           
initializer = mx.init.Xavier(rnd_type='gaussian', 
                             factor_type="in",
                             magnitude=2)  
model_prefix="check_point\\ssdh"
checkpoint = mx.callback.do_checkpoint(model_prefix)

eval_metric=MyAccuracy() 
 
#model.bind([("data",(batch_size,3,256,256))])
#model.init_params(initializer,
#                  arg_params= load_args,
#                  aux_params= load_aux_params,
#                  allow_missing = True)
#model.forward(trainIter.next(),is_train=True)
#model.backward()
    
model.fit(trainIter,
          begin_epoch=0,
          num_epoch=5,
          eval_data=valIter,
          eval_metric=eval_metric,
          optimizer=optimizer,
          initializer=initializer,
          arg_params= load_args,
          aux_params= load_aux_params,
          batch_end_callback = mx.callback.Speedometer(batch_size, 5),   
          allow_missing = True,  
          epoch_end_callback=checkpoint)



