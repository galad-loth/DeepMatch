# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 08:00:11 2017

@author: Fengjilan
"""
import numpy as npy
import mxnet as mx
import logging
import sys
from net_symbol.deep_compare_symbol import GetDeepCompareSymbol
from utils.data import GetUBCPatchDataIter
from utils.evaluate_metric import PNAccuracy


logging.basicConfig(level=logging.INFO)

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

def train_deep_compare():
    train_net=GetDeepCompareSymbol("2ch",True)
    train_model = mx.model.FeedForward(
             train_net,
             ctx=mx.context.gpu(),
             initializer=mx.initializer.Xavier(),
             num_epoch=10,
             learning_rate=0.01,
             momentum=0.9,
             lr_scheduler=mx.lr_scheduler.FactorScheduler(10000,0.9),
             wd=0.001)

    datadir="E:\\DevProj\\Datasets\\UBCPatch"
    dataset="liberty"
    gt_file="m50_100000_100000_0.txt"
    batch_size=50
    trainIter,valIter=GetUBCPatchDataIter(datadir, dataset,gt_file, 
                                          batch_size, "2ch",True,0.1)
    model_prefix="deep_compare"
    checkpoint = mx.callback.do_checkpoint(model_prefix) 

                                          
    train_model.fit(X=trainIter,
              eval_data=valIter,
              eval_metric=PNAccuracy,
              epoch_end_callback=checkpoint)

if __name__=="__main__":
    train_deep_compare()
    # test_deep_compare()
