# -*- codingL utf-8-*-
"""
Created on Tue Oct 07 10:10:15 2018
@author: galad-loth
"""
import numpy as npy
from matplotlib import pyplot as plt
import cv2
from cnn_desc import get_cnn_desc

img1=cv2.imread(r"D:\_Datasets\VGGAffine\ubc\img1.ppm",cv2.IMREAD_COLOR)
img2=cv2.imread(r"D:\_Datasets\VGGAffine\ubc\img2.ppm",cv2.IMREAD_COLOR)
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gap_width=20
black_gap=npy.zeros((img1.shape[0],gap_width),dtype=npy.uint8)
border_width=64
kpt_mask=npy.zeros(gray1.shape, dtype=npy.uint8)
kpt_mask[border_width:-border_width,border_width:-border_width]=1

sift = cv2.xfeatures2d.SIFT_create(nfeatures=200)
kpt1 = sift.detect(gray1,kpt_mask)
#kpt1, desc1 = sift.compute(gray1,kpt1)
kpt1, desc1=get_cnn_desc(gray1, kpt1)

kpt2 = sift.detect(gray2,kpt_mask)
#kpt2, desc2 = sift.compute(gray2,kpt2)
kpt2, desc2=get_cnn_desc(gray2, kpt1)

matcher = cv2.BFMatcher(cv2.NORM_L2SQR)
match_pairs = matcher.knnMatch(desc1,desc2,k=2)

good_matches=[]
for bm1,bm2 in match_pairs:
    if bm1.distance < 0.7*bm2.distance:
        good_matches.append(bm1)
        
good_matches=sorted(good_matches, key = lambda x:x.distance)

if len(good_matches)>10:
    pts_from = npy.float32([kpt1[bm.queryIdx].pt for bm in good_matches]).reshape(-1,1,2)
    pts_to = npy.float32([kpt2[bm.trainIdx].pt for bm in good_matches]).reshape(-1,1,2)
    mat_H, match_mask = cv2.findHomography(pts_from, pts_to, cv2.RANSAC,5.0)

imgcnb=npy.concatenate((gray1,black_gap,gray2),axis=1)

plt.figure(1,figsize=(15,6))
plt.imshow(imgcnb,cmap="gray")
idx=0
for bm in good_matches[:30]:
    if 1==match_mask[idx]:
        kpt_from=kpt1[bm.queryIdx]
        kpt_to=kpt2[bm.trainIdx]
        plt.plot(kpt_from.pt[0],kpt_from.pt[1],"rs",
                 markerfacecolor="none",markeredgecolor="r",markeredgewidth=2)
        plt.plot(kpt_to.pt[0]+img1.shape[1]+gap_width,kpt_to.pt[1],"bo",
                 markerfacecolor="none",markeredgecolor="b",markeredgewidth=2)
        plt.plot([kpt_from.pt[0],kpt_to.pt[0]+img1.shape[1]+gap_width],
                 [kpt_from.pt[1],kpt_to.pt[1]],"c-",linewidth=2)
    idx+=1
plt.axis("off")
