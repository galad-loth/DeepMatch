# -*- codingL utf-8-*-
"""
Created on Tue Sep 25 11:28:35 2018
@author: galad-loth
"""

import numpy as npy
import mxnet as mx

def pn_accuracy(gt_label, pred_val):
    """
    compute the accuracy for +1/-1 binary label
    """
    pred_label=npy.ones(pred_val.shape,dtype=npy.int8)
    pred_label[pred_val<0]=-1
    if len(gt_label.shape) == 1:
        gt_label = gt_label.reshape(gt_label.shape[0], 1)
    
    return npy.sum(pred_label == gt_label)*1.0/gt_label.size  


if __name__=="__main__":
    metric=mx.metric.create(pn_accuracy)