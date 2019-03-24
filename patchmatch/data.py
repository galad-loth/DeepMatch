# -*- codingL utf-8-*-
"""
Created on Tue Sep 25 11:28:35 2018
@author: galad-loth
"""

import numpy as npy
import mxnet as mx
import cv2
import os
   
class UBCPatchBatch(object):
    def __init__(self, data1, data2, label, align_type,pad=0):
        if align_type=='concat':
            self.data =[mx.nd.array(npy.concatenate((data1[:,npy.newaxis,:,:],
                                        data2[:,npy.newaxis,:,:]), axis=1))]
        elif align_type=="siam":
            self.data=[mx.nd.array(data1[:,npy.newaxis,:,:]),
                              mx.nd.array(data2[:,npy.newaxis,:,:])]
        self.label =[mx.nd.array(label)]
        self.pad = pad        
    
class UBCPatchDataIter(mx.io.DataIter):
    def __init__(self,data,pair_idx,label,batch_size,align_type):
        super(UBCPatchDataIter, self).__init__()
        self._data=data
        self._pair_idx = pair_idx
        self._label = label
        self.batch_size=batch_size
        self.batch_num=pair_idx.shape[0]/batch_size
        self.cur_batch=0
        self._align_type=align_type
        if align_type=='concat':
            self._provide_data=zip(["data"],[(batch_size, 2, 64, 64)])
            self._provide_label=[("loss_label",(batch_size,))]
        elif align_type=="siam":
            self._provide_data=zip(["data1","data2"],                                   
                                   [(batch_size, 1, 64, 64),(batch_size, 1, 64, 64)])
            self._provide_label=[("loss_label",(batch_size,))]
        
    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0        

    def __next__(self):
        return self.next()
  
    @property
    def provide_data(self):      
        return self._provide_data

    @property
    def provide_label(self):       
        return self._provide_label
    
    def next(self):
        if self.cur_batch < self.batch_num:
            idx_start=self.cur_batch*self.batch_size
            idx_end=(self.cur_batch+1)*self.batch_size            
            batch_label=self._label[idx_start:idx_end]
            batch_data1=self._data[self._pair_idx[idx_start:idx_end,0],:,:]  
            batch_data2=self._data[self._pair_idx[idx_start:idx_end,1],:,:] 
            batch_data1=(batch_data1/128.0-1).astype(npy.float32)
            batch_data2=(batch_data2/128.0-1).astype(npy.float32)
            self.cur_batch += 1
            return UBCPatchBatch(batch_data1, batch_data2,
                                 batch_label,self._align_type)
        else:
            raise StopIteration

def load_UBC_patch_data(datadir,dataname):
    patch_width=64
    patch_height=64    
    imgdir=os.path.join(datadir,dataname,"images")
    file_list=os.listdir(imgdir)
    num_files=len(file_list)
    data_all=npy.zeros((num_files*256,patch_height,patch_width),
                       dtype=npy.uint8)
    num_patch=0        
    for f in file_list:
        print("Reading {}".format(f))
        img=cv2.imread(os.path.join(imgdir,f), cv2.IMREAD_GRAYSCALE)
        for iy in npy.arange(16):
            for ix in npy.arange(16):
                data_all[num_patch,:,:]=img[iy*64:(iy+1)*64,ix*64:(ix+1)*64]
                num_patch+=1
    return data_all 

def get_UBC_patch_dataiter(
        datadir, dataname, gt_file, batch_size, align_type,
        flag_train=False, val_ratio=0):
    gt_info=npy.loadtxt(os.path.join(datadir, dataname, gt_file),
                        dtype=npy.int32)
    num_gt_data=gt_info.shape[0]
    pair_idx=gt_info[:,[0,3]]
    gt_label=npy.ones(num_gt_data,dtype=npy.int8)
    if align_type=='concat':
        gt_label[gt_info[:,1]!=gt_info[:,4]]=-1 #unmatch pairs have label -1
    elif align_type=="siam":
        gt_label[gt_info[:,1]!=gt_info[:,4]]=0 #unmatch pairs have label 0
    
    patchdata=load_UBC_patch_data(datadir,dataname)
    
    if flag_train:
        num_val=npy.int32(num_gt_data*val_ratio)
        num_train=num_gt_data-num_val
        trainIter=UBCPatchDataIter(patchdata,
                                   pair_idx[:num_train,:],
                                   gt_label[:num_train],
                                   batch_size,
                                   align_type)
        valIter = UBCPatchDataIter(patchdata,
                                   pair_idx[num_train:,:],
                                   gt_label[num_train:],
                                   batch_size,
                                   align_type)
        return trainIter,valIter
    else:
        testIter= UBCPatchDataIter(patchdata,
                                   pair_idx,
                                   gt_label,
                                   batch_size,
                                   align_type)
        return testIter

if __name__=="__main__":
   datadir="D:\\_Datasets\\UBCPatch"
   dataset="liberty"
   gt_file="m50_10000_10000_0.txt"
   batch_size=100
   align_type="siam"
   flag_train=True
   trainIter,valIter=get_UBC_patch_dataiter(datadir, dataset, gt_file, 
                                         batch_size, align_type, flag_train, 0.1)
   data=trainIter.next()
   img_data0=data.data[0].asnumpy()
   img_data1=data.data[1].asnumpy()
   label=data.label[0].asnumpy()
   