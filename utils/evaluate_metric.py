import numpy as npy
import mxnet as mx

def PNAccuracy(gt_label, pred_val):
#    print pred_val
    pred_label=npy.ones(pred_val.shape,dtype=npy.int8)
    pred_label[pred_val<0]=-1
    if len(gt_label.shape) == 1:
        gt_label = gt_label.reshape(gt_label.shape[0], 1)
    
    return npy.sum(pred_label == gt_label)*1.0/gt_label.size

