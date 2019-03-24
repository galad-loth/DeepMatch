# -*- coding utf-8-*-
"""
Created on Tue Nov 23 10:15:35 2018
@author: galad-loth
"""

import mxnet as mx
import logging
import sys
from hash_net import get_ssdh_symbol
from evaluate_metric import MyAccuracy
from data import get_img_class_iter

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

def train_ssdh():
    pretrain_model=(r'D:\Pretrained\mxnet\Inception-BN')
    load_net, load_arg_params, load_aux_params = \
        mx.model.load_checkpoint(pretrain_model, 126)
    new_net, load_args=get_ssdh_symbol(load_net,load_arg_params,512,45)
    
    batch_size=10
    datadir=r"D:\Jilan_Work\DevProj\_Datasets\NWPU-RESISC45\images"
    trainIter,valIter, cls_dict=get_img_class_iter(datadir,(batch_size,3,256,256),True,0.4)
    
    model = mx.mod.Module(symbol= new_net, context= mx.gpu())
    
    optimizer = mx.optimizer.create('sgd',
                                    rescale_grad=1.0/batch_size,
                                    learning_rate =0.01,
                                    momentum = 0.9,
                                    wd = 0.0005,
                                    lr_scheduler=mx.lr_scheduler.FactorScheduler(250,0.9))
    
    new_net_args=new_net.list_arguments()
    lr_scale={}
    for arg_name in new_net_args:
        if "ssdh" in arg_name:
            lr_scale[arg_name] = 10
    optimizer.set_lr_mult(lr_scale)
    
    initializer = mx.init.Xavier(rnd_type='gaussian', 
                                 factor_type="in",
                                 magnitude=2)  
    model_prefix="checkpoint\\ssdh"
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    
    eval_metric=MyAccuracy() 
    
    model.fit(trainIter,
              begin_epoch=0,
              num_epoch=2,
              eval_data=valIter,
              eval_metric=eval_metric,
              optimizer=optimizer,
              initializer=initializer,
              arg_params= load_args,
              aux_params= load_aux_params,
              batch_end_callback = mx.callback.Speedometer(batch_size, 5),   
              allow_missing = True,  
              epoch_end_callback=checkpoint)

if __name__=="__main__":
    train_ssdh()
    # test_deep_compare()