import numpy as npy
import mxnet as mx
import cv2
import os
   
class UBCPatchBatch(object):
    def __init__(self, data1, data2, label, net_type,pad=0):
        if net_type=='2ch':
            self.data =[mx.nd.array(npy.concatenate((data1[:,npy.newaxis,:,:],
                                         data2[:,npy.newaxis,:,:]), axis=1))]
        elif net_type=="siam":
            self.data=[mx.nd.array(data1[:,npy.newaxis,:,:]),
                       mx.nd.array(data2[:,npy.newaxis,:,:])]
        self.label =[mx.nd.array(label)]
        self.pad = pad        
    
class UBCPatchDataIter(mx.io.DataIter):
    def __init__(self,data,pair_idx,label,batch_size,net_type):
        super(UBCPatchDataIter, self).__init__()
        self._data=data
        self._pair_idx = pair_idx
        self._label = label
        self.batch_size=batch_size
        self.batch_num=pair_idx.shape[0]/batch_size
        self.cur_batch=0
        self._net_type=net_type
        if net_type=='2ch':
            self._provide_data=zip(["data"],[(batch_size,2,64,64)])
            self._provide_label=[("loss_label",(batch_size,))]
        elif net_type=="siam":
            self._provide_data=zip(["data1","data2"],                                   
                                   [(batch_size,1,64,64),(batch_size,1,64,64)])
            self._provide_label=("loss_label",(batch_size,))
        
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
                                 batch_label,self._net_type)
        else:
            raise StopIteration

def LoadUBCPatchData(datadir,dataname):
    patch_width=64
    patch_height=64    
    imgdir=os.path.join(datadir,dataname,"images")
    file_list=os.listdir(imgdir)
    num_files=len(file_list)
    data_all=npy.zeros((num_files*256,patch_height,patch_width),dtype=npy.uint8)
    num_patch=0        
    for f in file_list:
        print "Reading "+f
        img=cv2.imread(os.path.join(imgdir,f), cv2.IMREAD_GRAYSCALE)
        for iy in npy.arange(16):
            for ix in npy.arange(16):
                data_all[num_patch,:,:]=img[iy*64:(iy+1)*64,ix*64:(ix+1)*64]
                num_patch+=1
    return data_all 

  
def GetUBCPatchDataIter(datadir,dataname,gt_file,batch_size,net_type,
                        flag_train=False, val_ratio=0):
    
    gt_info=npy.loadtxt(os.path.join(datadir, dataname, gt_file),dtype=npy.int32)
    num_gt_data=gt_info.shape[0]
    pair_idx=gt_info[:,[0,3]]
    gt_label=npy.ones(num_gt_data,dtype=npy.int8)
    gt_label[gt_info[:,1]!=gt_info[:,4]]=-1
    
    patchdata=LoadUBCPatchData(datadir,dataname)
    
    if flag_train:
        num_val=npy.int32(num_gt_data*val_ratio)
        num_train=num_gt_data-num_val
        trainIter=UBCPatchDataIter(patchdata,
                                   pair_idx[:num_train,:],
                                   gt_label[:num_train],
                                   batch_size,
                                   net_type)
        valIter = UBCPatchDataIter(patchdata,
                                   pair_idx[num_train:,:],
                                   gt_label[num_train:],
                                   batch_size,
                                   net_type)
        return trainIter,valIter
    else:
        testIter= UBCPatchDataIter(patchdata,
                                   pair_idx,
                                   gt_label,
                                   batch_size,
                                   net_type)
        return testIter
 
class ImgClassBatch(object):
    def __init__(self, data, label, pad=0):
        self.data =[mx.nd.array(data)]
        self.label =[mx.nd.array(label)]
        self.pad = pad        
    
class ImgClassIter(mx.io.DataIter):
    def __init__(self,datadir,img_list,label_list, data_shape, num_class):
        super(ImgClassIter, self).__init__()
        self._datadir=datadir
        self._img_list = img_list
        self._label_list = label_list
        self._num_class=num_class
        self.batch_size=data_shape[0]
        assert len(img_list)==len(label_list)
        self.batch_num=len(img_list)/data_shape[0]
        self.cur_batch=0
        self._provide_data=zip(["data"],[data_shape])
        self._provide_label=[("softmax_label",(data_shape[0],num_class))]
        self.img_mean=npy.reshape(npy.array([128,128,128]),(1,1,3))
               
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
            data_shape=self._provide_data[0][1]
            batch_data=npy.zeros(data_shape, dtype=npy.float32)
            batch_label=npy.zeros((data_shape[0], self._num_class), dtype=npy.int32)
            for idx in npy.arange(0, self.batch_size):
                idx_start=self.batch_size*self.cur_batch 
                img=cv2.imread(os.path.join(self._datadir,self._img_list[idx_start+idx]),
                               cv2.IMREAD_COLOR)
                img=(img-self.img_mean)/96.0                
                img = npy.swapaxes(img, 0, 2)
                img = npy.swapaxes(img, 1, 2)  # (c, h, w)
                
                batch_data[idx,:,:,:]=img
                batch_label[idx, self._label_list[idx_start+idx]]=1            
            self.cur_batch += 1
            return ImgClassBatch(batch_data,batch_label,0)
        else:
            raise StopIteration
            
            
def GetImgClassIter(datadir, data_shape, is_train=False, val_ratio=0):
    cls_list=os.listdir(datadir)
    if is_train:
        train_img_list=[]
        train_label_list=[]
        val_img_list=[]
        val_label_list=[]
        idx_cls=0
        for cls in cls_list:
            img_list=os.listdir(os.path.join(datadir,cls))
            num_img=len(img_list)
            num_val=int(num_img*val_ratio)
            idx_rand=npy.random.permutation(num_img)
            val_img_list+=[os.path.join(cls,img_list[idx_rand[i]]) for i in xrange(num_val)]
            val_label_list+=[idx_cls for i in xrange(num_val)]
            train_img_list+=[os.path.join(cls,img_list[idx_rand[num_val+i]]) \
                             for i in xrange(num_img-num_val)]
            train_label_list+=[idx_cls for i in xrange(num_img-num_val)]
            idx_cls+=1
        idx_rand=npy.random.permutation(len(train_img_list))
        train_img_list=[train_img_list[i] for i in idx_rand]
        train_label_list=[train_label_list[i] for i in idx_rand]
        trainIter= ImgClassIter(datadir,
                                train_img_list,
                                train_label_list,
                                data_shape,
                                 len(cls_list))  
        valIter = ImgClassIter(datadir,
                                val_img_list,
                                val_label_list,
                                data_shape,
                                len(cls_list)) 
        return trainIter, valIter,zip(range(len(cls_list)),cls_list)
    else:
        img_list=[]
        label_list=[]
        idx_cls=0
        for cls in cls_list:
            img_list_temp=os.listdir(os.path.join(datadir,cls))
            num_img=len(img_list)
            img_list+=[os.path.join(cls,img_list_temp[i]) for i in xrange(num_img)]
            label_list+=[idx_cls for i in xrange(num_img)]
            idx_cls+=1
        testIter= ImgClassIter(datadir,
                                img_list,
                                label_list,
                                data_shape,
                                 len(cls_list)) 
        return testIter,cls_list ,zip(range(len(cls_list)),cls_list)  
    
    
if __name__=="__main__":
#    datadir="E:\\DevProj\\Datasets\\UBCPatch"
#    dataset="liberty"
#    gt_file="m50_10000_10000_0.txt"
#    batch_size=100
#    trainIter,valIter=GetUBCPatchDataIter(datadir, dataset,gt_file, 
#                                          batch_size, "2ch",True,0.1)
    datadir="E:\\DevProj\\Datasets\\NWPU-RESISC45\\images"
    trainIter,valIter, cls_dict=GetImgClassIter(datadir,(50,3,256,256),True,0.2)