import numpy as npy
import mxnet as mx

def pn_accuracy(gt_label, pred_val):
    pred_label=npy.ones(pred_val.shape,dtype=npy.int8)
    pred_label[pred_val<0]=-1
    if len(gt_label.shape) == 1:
        gt_label = gt_label.reshape(gt_label.shape[0], 1)
    
    return npy.sum(pred_label == gt_label)*1.0/gt_label.size   
    
    
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
        

