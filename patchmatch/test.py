import numpy as npy
from matplotlib import pyplot as plt
import cv2
import mxnet as mx

ctx=mx.cpu()
load_net, load_arg_params, load_aux_params = mx.model.load_checkpoint('checkpoint\\matchnet', 13)
all_layers=load_net.get_internals()
net=all_layers["feat1_bottleneck_output"]
new_args = dict({k:load_arg_params[k] for k in load_arg_params
                 if 'fc' not in k})
mod = mx.mod.Module(symbol=net, context=ctx, data_names=['data1'],label_names=None)
mod.bind(for_training=False, data_shapes=[('data1', (1,1,64,64))], 
         label_shapes=mod._label_shapes)
mod.set_params(new_args, load_aux_params, allow_missing=True)
