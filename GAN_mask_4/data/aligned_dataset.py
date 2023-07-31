import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json
import numpy as np


class AlignedDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, 'images', opt.phase) #phase is train/test/...
        self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)

        #self.AB_paths, self.bbox_paths = sorted(make_dataset(self.dir_AB, self.dir_bbox))
        self.AB_paths, self.bbox_paths = make_dataset(self.dir_AB, self.dir_bbox)
        self.AB_paths = sorted(self.AB_paths)
        self.bbox_paths = sorted(self.bbox_paths)

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(), # convert image to tensor
                          transforms.Normalize(((0.5, ) * opt.input_nc),
                                               ((0.5, ) * opt.input_nc))] # list that holds the defined transformations. In this case, it contains the ToTensor() transformation and the Normalize() transformation.

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
        AB_path = self.AB_paths[index]
        bbox_path = self.bbox_paths[index]

        w_total = self.opt.loadSize * 2
        w = int(w_total / 2)
        h = self.opt.loadSize
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        #print(w_offset,h_offset)

        bbox = json.load(open(bbox_path))
        AB = Image.open(AB_path)

        size_x = AB.size[0]
        size_y = AB.size[1]
        h2 = self.opt.fineSize
        w_total2 = h2*2
        bbox = {"y":int(bbox["y"]*h2/size_y),"x" : int(bbox["x"]*w_total2/size_x), "w" : int(bbox["w"]*w_total2/size_x), "h" : int(bbox["h"]*h2/size_y)}
        bbox_x = max(int((bbox['x']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_y = max(int((bbox['y']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_w = max(int((bbox['w']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_h = max(int((bbox['h']/self.opt.fineSize)*self.opt.loadSize), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset or bbox_h <= h_offset or bbox_w <= w_offset or bbox_y >= h_offset + size_y or bbox_x >= w_offset +size_x/2 or bbox_h >= h_offset +size_y or bbox_w >= w_offset + size_x/2  :
            AB = AB.resize((self.opt.fineSize * 2, self.opt.fineSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, :self.opt.fineSize,
                :self.opt.fineSize]
            B = AB[:, :self.opt.fineSize,
                self.opt.fineSize:2*self.opt.fineSize]
            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
            #bbox_size = [bbox[3]-bbox[0],bbox[2]-bbox[1]]
            #print("1:",bbox_size)
        else:

            if self.opt.mask :
                img = Image.fromarray(np.array(AB)[:,:,0:3])
                mask = Image.fromarray(np.array(AB)[:,:,3])
                img = img.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
                mask = mask.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)

                AB = Image.fromarray(np.concatenate((np.array(img), np.array(mask)[..., np.newaxis]), axis=2))
            else :
                AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)

            AB = self.transform(AB)
            A = AB[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
            
            B = AB[:, h_offset:h_offset + self.opt.fineSize,
                w + w_offset:w + w_offset + self.opt.fineSize]
            bbox = [bbox_y-h_offset, bbox_x-w_offset, bbox_w-w_offset, bbox_h-h_offset]
        

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
