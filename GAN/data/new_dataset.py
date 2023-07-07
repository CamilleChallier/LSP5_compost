import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json
import cv2
import numpy as np


class NewDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def initialize(self, opt):
        """
        Create a list of all paths of the images and set the transfomations of these images
        """

        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot, 'images', opt.phase, 'real') #phase is train/test/...
        #self.dir_B = os.path.join(opt.dataroot, 'images', opt.phase, 'noise')
        self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)

        #self.AB_paths, self.bbox_paths = sorted(make_dataset(self.dir_AB, self.dir_bbox))
        self.A_paths, self.bbox_paths = make_dataset(self.dir_A, self.dir_bbox)
        self.A_paths = sorted(self.A_paths)
        self.bbox_paths = sorted(self.bbox_paths)

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(), # convert image to tensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))] # list that holds the defined transformations. In this case, it contains the ToTensor() transformation and the Normalize() transformation.

        self.transform = transforms.Compose(transform_list) #applies each transformation in the list to the data in the order they are defined

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        #load data
        A_path = self.A_paths[index]
        bbox_path = self.bbox_paths[index]
        A = Image.open(A_path).convert('RGB')
        bbox = json.load(open(bbox_path))

        #add noise to image
        image = cv2.imread(A_path,1)      
        roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 
        noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
        image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] = noise
        B_path = A_path.replace('real', 'noise')
        cv2.imwrite(B_path, image)
        B = Image.open(B_path).convert('RGB')

        #compute offset
        h = self.opt.loadSize
        h2 = self.opt.fineSize
        w_offset = random.randint(0, max(0, h - h2 - 1))
        h_offset = random.randint(0, max(0, h - h2 - 1))
        
        #resize
        size_x = A.size[0]
        size_y = A.size[1]
        bbox = {"y":int(bbox["y"]*h2/size_y),"x" : int(bbox["x"]*h2/size_x), "w" : int(bbox["w"]*h2/size_x), "h" : int(bbox["h"]*h2/size_y)}
        bbox_x = max(int((bbox['x']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_y = max(int((bbox['y']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_w = max(int((bbox['w']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_h = max(int((bbox['h']/self.opt.fineSize)*self.opt.loadSize), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset:
        #AB = Image.open(AB_path).convert('RGB')
            A = A.resize((self.opt.fineSize * 2, self.opt.fineSize), Image.BICUBIC)
            A = self.transform(A)
            B = B.resize((self.opt.fineSize * 2, self.opt.fineSize), Image.BICUBIC)
            B = self.transform(B)
            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]

        else:
            A = A.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            A = self.transform(A)
            B = B.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            B = self.transform(B)
            A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
            
            B = B[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
            
            bbox = [bbox_y-h_offset, bbox_x-w_offset, bbox_w-w_offset, bbox_h-h_offset]
        

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            #print A.size(2)
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
        # print(bbox)
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def name(self):
        return 'NewDataset'
