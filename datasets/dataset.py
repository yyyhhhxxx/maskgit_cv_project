import cv2 as cv
import os
import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

#以下实现不对，只是为了方便理解

# class TinyImageNet(Dataset):
#     def __init__(self, root_dir, filelist) -> None:
#         super().__init__()
#         self.root = root_dir  # data directory
#         self.filelist = self.load_files(filelist)  # read filelist

#     def __getitem__(self, index):
#         # get data
#         data = self.transform(self.filelist[index])
#         return data
        
    
#     def load_files(self, filelist):
#         # load file list
#         filelist = [os.path.join(self.root, line.strip()) for line in open(filelist, 'r')]
#         return filelist
        

#     def transform(self, data):
#         # processing data
#         data = cv.imread(data)
#         data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
#         data = cv.resize(data, (64, 64))
#         data = data / 255.0
#         data = np.transpose(data, (2, 0, 1))
#         data = torch.from_numpy(data).float()
#         return data
        

#     def __len__(self):
#         return len(self.filelist)
