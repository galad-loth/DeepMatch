import numpy as npy
import mxnet as mx
import logging
from net_symbol.deep_compare_symbol import GetDeepCompareSymbol
from utils.data import GetUBCPatchDataIter


logging.basicConfig(level=logging.INFO)

batch_size=100
train_net=GetDeepCompareSymbol("2ch",True)
model = mx.model.FeedForward(
         train_net,
         ctx=mx.context.gpu(),
         num_epoch=10,
         learning_rate=0.001,
         momentum=0.1,
         wd=0.01)

datadir="E:\\DevProj\\Datasets\\UBCPatch"
dataset="liberty"
gt_file="m50_2000_2000_0.txt"
batch_size=50
trainIter,valIter=GetUBCPatchDataIter(datadir, dataset,gt_file, 
                                          batch_size, "2ch",True,0.1)
model.fit(X=trainIter,
          eval_data=valIter,
          eval_metric='acc')


