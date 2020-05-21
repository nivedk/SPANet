import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from randomcrop import RandomRotation,RandomResizedCrop,RandomHorizontallyFlip,RandomVerticallyFlip
import PIL.Image as Image

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.mat_files = open(self.dataset,'r').readlines()
        self.file_num = len(self.mat_files)
        self.rc = RandomResizedCrop(256)

    def __len__(self):
        return self.file_num * 100

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        #gt_file = file_name.split(' ')[1][:-1]
        #img_file = file_name.split(' ')[0]

        a = cv2.imread(file_name)
        print(file_name)
        #B = cv2.imread(gt_file)

        s = a.shape
        #print(s)
        x = int(s[1]/2)
        #print(x)
        a1 = a[:,:x,:]
        a2 = a[:,x:,:]       

        O = Image.fromarray(a1)
        B = Image.fromarray(a2)

        O,B = self.rc(O,B)
        O,B = np.array(O),np.array(B)

        M = np.clip((O-B).sum(axis=2),0,1).astype(np.float32)
        O = np.transpose(O.astype(np.float32) / 255, (2, 0, 1))
        B = np.transpose(B.astype(np.float32) / 255, (2, 0, 1)) 

        sample = {'O': O, 'B': B,'M':M}

        return sample



class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = name
        self.mat_files = open(self.root_dir,'r').readlines()

        self.file_num = len(self.mat_files)
        
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        #gt_file = "." + file_name.split(' ')[1][:-1]
        #img_file = "." + file_name.split(' ')[0]
        
        #O = cv2.imread(img_file)
        #B = cv2.imread(gt_file)
        file_name = file_name[:-1]
        print(file_name)
        print("hello")
        a = cv2.imread(file_name)
        #a = Image.open(file_name)
        #a = np.array(a)
        print(a.shape)
        #B = cv2.imread(gt_file)
        print(type(a))
        s = a.shape
        #print(s)
        x = int(s[1]/2)
        #print(x)
        a1 = a[:,:x,:]
        a2 = a[:,x:,:]       

        O = Image.fromarray(a1)
        B = Image.fromarray(a2)


        O = np.transpose(O, (2, 0, 1)).astype(np.float32) / 255.0 
        B = np.transpose(B, (2, 0, 1)).astype(np.float32) / 255.0 

        sample = {'O': O,'B':B,'M':O}

        return sample
