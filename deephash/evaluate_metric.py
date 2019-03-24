# -*- coding utf-8-*-
"""
Created on Tue Nov 23 10:15:35 2018
@author: galad-loth
"""
import numpy as npy
import mxnet as mx
    
class MyAccuracy(mx.metric.EvalMetric):
    def __init__(self):
        super(MyAccuracy, self).__init__('myacc')
    def update(self, labels, preds):
        pred_label = mx.ndarray.argmax_channel(preds[0])
        pred_label = pred_label.asnumpy().astype('int32')
        gt_label = labels[0].asnumpy().astype('int32')
        
        if pred_label.shape != gt_label.shape:
            raise ValueError("Shape of pred_label {} does not match shape of "
                         "gt_label {}".format(pred_label.shape, gt_label.shape))
        
        self.sum_metric += (pred_label.flat == gt_label.flat).sum()
        self.num_inst += len(pred_label.flat)
        

if __name__ == "__main__":
    myacc=MyAccuracy()
    predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    labels   = [mx.nd.array([0, 1, 1])]
    myacc.update(labels = labels, preds = predicts) 
    print(myacc.get())

