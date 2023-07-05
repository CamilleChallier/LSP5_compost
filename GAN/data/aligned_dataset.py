import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json


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
        AB_path = self.AB_paths[index]
        #print(AB_path)
        bbox_path = self.bbox_paths[index]
        #print(bbox_path)

        w_total = self.opt.loadSize * 2
        w = int(w_total / 2)
        h = self.opt.loadSize
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        bbox = json.load(open(bbox_path))
        #bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        #print(bbox['y'], bbox['x'], bbox['w'], bbox['h'])
        bbox_x = max(int((bbox['x']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_y = max(int((bbox['y']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_w = max(int((bbox['w']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_h = max(int((bbox['h']/self.opt.fineSize)*self.opt.loadSize), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset:
            AB = Image.open(AB_path).convert('RGB')
            AB = AB.resize((self.opt.fineSize * 2, self.opt.fineSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, :self.opt.fineSize,
               :self.opt.fineSize]
            B = AB[:, :self.opt.fineSize,
                self.opt.fineSize:2*self.opt.fineSize]
            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        else:
            AB = Image.open(AB_path).convert('RGB')
            AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
            B = AB[:, h_offset:h_offset + self.opt.fineSize,
                w + w_offset:w + w_offset + self.opt.fineSize]
            bbox = [bbox_y-h_offset, bbox_x-w_offset, bbox_w, bbox_h]
        # print(bbox)
        

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            #print A.size(2)
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
        # print(bbox)
        #print(A.size())
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'