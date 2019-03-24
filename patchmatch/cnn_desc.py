# -*- codingL utf-8-*-
"""
Created on Tue Oct 07 10:10:15 2018
@author: galad-loth
"""

import numpy as npy
from matplotlib import pyplot as plt
import mxnet as mx
import cv2
from collections import namedtuple

def extract_oriented_patches(img, kpt, patch_size):
    resize_factor=8
    img1=cv2.resize(img,(0,0),fx=8,fy=8)
    [img_height, img_width]=img1.shape
    kpt_num=len(kpt)
    patch_data=npy.zeros((kpt_num, patch_size*patch_size), dtype=npy.float32)
    valid_flag=npy.ones(kpt_num,dtype=npy.bool)
    sample_range=resize_factor*(npy.arange(patch_size)-patch_size//2)
    grid_x, grid_y=npy.meshgrid(sample_range,sample_range)
    sample_grid=npy.stack([grid_x.flatten(), grid_y.flatten()])
    for kpt_idx in npy.arange(kpt_num):
        pt=resize_factor*npy.array(kpt[kpt_idx].pt)
        angle=kpt[kpt_idx].angle/180*npy.pi
        rot_mat=npy.array([[npy.cos(angle), -npy.sin(angle)],[npy.sin(angle), npy.cos(angle)]])
        sample_grid_rot=npy.dot(rot_mat,sample_grid)+pt[:,npy.newaxis]
        sample_grid_rot=sample_grid_rot.astype(npy.int32)
        for idx in npy.arange(patch_size*patch_size):
            px=sample_grid_rot[0,idx]
            py=sample_grid_rot[1,idx]
            if px < 0 or py < 0 or px > img_width-1 or py > img_height-1:
                valid_flag[kpt_idx]=False
                break
            patch_data[kpt_idx, idx]=img1[py,px]
    patch_data=patch_data.reshape((kpt_num, patch_size, patch_size))
    patch_data=(patch_data/128.0-1).astype(npy.float32)
    return patch_data, valid_flag
        
Batch = namedtuple('Batch', ['data'])
def extrac_cnn_desc(patch_data):
    load_net, load_arg_params, load_aux_params = mx.model.load_checkpoint('checkpoint\\matchnet', 13)
    all_layers=load_net.get_internals()
    net=all_layers["feat1_bottleneck_output"]
    new_args = dict({k:load_arg_params[k] for k in load_arg_params
                     if 'fc' not in k})
    mod = mx.mod.Module(symbol=net, context=mx.cpu(), data_names=['data1'],label_names=None)
    mod.bind(for_training=False, data_shapes=[('data1', (1,1,64,64))], 
             label_shapes=mod._label_shapes)
    mod.set_params(new_args, load_aux_params, allow_missing=True)
    patch_data1=Batch([mx.nd.array(patch_data[:,npy.newaxis,:,:])])
    mod.forward(patch_data1)
    desc=mod.get_outputs()[0].asnumpy()
    return desc



def get_cnn_desc(img, kpt):
    patch_size=64
    patch_data,valid_flag=extract_oriented_patches(img, kpt, patch_size) 
    patch_data=patch_data[valid_flag,:,:]
    kpt_ret=[k for [k,f] in zip(kpt, valid_flag) if f>0]
#    mod=init_cnn_mod(len(kpt_ret),"matchnet")
#    desc=mod.forward(data=patch_data)
    desc=extrac_cnn_desc(patch_data)
    return kpt_ret, desc
    
if __name__ == "__main__":
    img1=cv2.imread(r"D:\_Datasets\VGGAffine\ubc\img1.ppm",cv2.IMREAD_COLOR)
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    kpt_mask=npy.zeros(gray1.shape, dtype=npy.uint8)
    border_width=64
    kpt_mask[border_width:-border_width,border_width:-border_width]=1
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=200)
    kpt1 = sift.detect(gray1,kpt_mask)
    
    kpt_ret, desc=get_cnn_desc(gray1, kpt1)
    
    
#    patch_data, valid_flag=extract_oriented_patches(gray1, kpt1, 64)
    
#    idx_pt=100
#    pt=npy.array(kpt1[idx_pt].pt).astype(npy.int32)
#    gray1[(pt[1]-5):(pt[1]+5),(pt[0]-5):(pt[0]+5)]=255
#    plt.figure(1)
#    plt.imshow(gray1)
#    plt.figure(2)
#    plt.imshow(patch_data[idx_pt,:].reshape((64,64)))
#    plt.title("angle={}".format(kpt1[idx_pt].angle))
    