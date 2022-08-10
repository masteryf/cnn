import cv2
import os
import numpy as np

for img in os.listdir('dataset-imgs/train'):
    with open('dataset/train/label','a')as lab:
        head,sep,tail=img.partition('-')
        #newstring = ''.join([i for i in head if not i.isdigit()])
        lab.write(head+'\n')
    print(img)
    frame=cv2.imread('dataset-imgs/train/'+img,0)
    frame=cv2.resize(frame,(128,128),interpolation=cv2.INTER_AREA)
    with open('dataset/train/dataset','ab') as file:
        np.savetxt(file, frame, fmt='%d', delimiter='\t')