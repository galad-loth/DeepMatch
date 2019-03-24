import numpy as npy
import mxnet as mx
import cv2
import os


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
        self.data_shape=data_shape
        self.batch_size=data_shape[0]
        assert len(img_list)==len(label_list)
        self.batch_num=len(img_list)/data_shape[0]
        self.cur_batch=0
        self._provide_data=zip(["data"],[data_shape])
        self._provide_label=[("softmax_label",(data_shape[0],))]
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
            batch_data=npy.zeros(self.data_shape, dtype=npy.float32)
#            batch_label=npy.zeros((data_shape[0], self._num_class), dtype=npy.int32)
            idx_start=self.batch_size*self.cur_batch 
            for idx in npy.arange(0, self.batch_size):                
                img=cv2.imread(os.path.join(self._datadir,self._img_list[idx_start+idx]),
                               cv2.IMREAD_COLOR)
                img=(img-self.img_mean)/96.0                
                img = npy.swapaxes(img, 0, 2) #(c ,w, h)
                img = npy.swapaxes(img, 1, 2)  # (c, h, w)
                
                batch_data[idx,:,:,:]=img
#                batch_label[idx, self._label_list[idx_start+idx]]=1            
            batch_label=npy.array(self._label_list[idx_start:idx_start+self.batch_size])
            self.cur_batch += 1
            return ImgClassBatch(batch_data,batch_label,0)
        else:
            raise StopIteration
            
            
def get_img_class_iter(datadir, data_shape, is_train=False, val_ratio=0):
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
            val_img_list+=[os.path.join(cls,img_list[idx_rand[i]]) for i in npy.arange(num_val)]
            val_label_list+=[idx_cls for i in npy.arange(num_val)]
            train_img_list+=[os.path.join(cls,img_list[idx_rand[num_val+i]]) \
                             for i in npy.arange(num_img-num_val)]
            train_label_list+=[idx_cls for i in npy.arange(num_img-num_val)]
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
            img_list+=[os.path.join(cls,img_list_temp[i]) for i in npy.arange(num_img)]
            label_list+=[idx_cls for i in npy.arange(num_img)]
            idx_cls+=1
        testIter= ImgClassIter(datadir,
                                img_list,
                                label_list,
                                data_shape,
                                 len(cls_list)) 
        return testIter,cls_list ,zip(range(len(cls_list)),cls_list)  
    
    
if __name__=="__main__":
    datadir=r"D:\_Datasets\NWPU-RESISC45\images"
    trainIter,valIter, cls_dict=get_img_class_iter(datadir,(50,3,256,256),True,0.2)