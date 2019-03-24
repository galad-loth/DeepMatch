# -*- coding utf-8-*-
"""
Created on Tue Nov 23 10:15:35 2018
@author: galad-loth
"""
import os
import numpy as npy
import mxnet as mx
import cv2
import hash_net
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])
img_mean=npy.reshape(npy.array([128,128,128]),(1,1,3))

data_dir=r"D:\_Datasets\NWPU-RESISC45\images"
temp_data_dir=r"data"

def get_hash_code():
    load_net, load_arg_params, load_aux_params = mx.model.load_checkpoint('checkpoint\\ssdh', 2)
    all_layers=load_net.get_internals()
    net=all_layers["sigmoid_ssdh_output"]
    new_args = dict({k:load_arg_params[k] for k in load_arg_params
                         if 'fc_ssdh_2' not in k})
    mod = mx.mod.Module(symbol=net, context=mx.gpu(), data_names=['data'],label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,256,256))], 
                 label_shapes=mod._label_shapes)
    mod.set_params(new_args, load_aux_params, allow_missing=True)
        
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)
    
    for class_dir in os.listdir(data_dir):
        print("processing class {}".format(class_dir))
        save_file=os.path.join(temp_data_dir,class_dir+".csv")
        image_list=os.listdir(os.path.join(data_dir, class_dir))
        image_num=len(image_list)
        save_data=npy.zeros((image_num, mod.output_shapes[0][1][1]), dtype=npy.uint8)
        for idx in npy.arange(image_num):
            image_file=image_list[idx]
            img=cv2.imread(os.path.join(data_dir, class_dir, image_file), cv2.IMREAD_COLOR)
            img=(img-img_mean)/96.0                
            img = npy.swapaxes(img, 0, 2) #(c ,w, h)
            img = npy.swapaxes(img, 1, 2)  # (c, h, w)
            imput_data=Batch([mx.nd.array(img[npy.newaxis,:,:,:])])
            mod.forward(imput_data)
            output_data=mod.get_outputs()[0].asnumpy()
            hash_code=(output_data>0.5).astype(npy.uint8)
            save_data[idx,:]=hash_code
        npy.savetxt(save_file, save_data, delimiter=',', fmt="%d")
        
    
def test_img_retrival():
    pass
            
    
if __name__ == "__main__":
#    get_hash_code()
    test_img_retrival()
    

