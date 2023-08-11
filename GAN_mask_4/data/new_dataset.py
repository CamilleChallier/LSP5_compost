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

def replace_plastics_with_noise(generator,image,bbox,mask, noise_fct):
    # Iterate over the bounding boxes
    image = np.array(image)

    roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 

    if mask : 
        channel_nb = 4
    else : 
        channel_nb =3
        
    if (noise_fct == "b_w") :
        noise = generator.integers(low=0, high=2, size=[roi.shape[0],roi.shape[1]], dtype=np.int64)
        noise = np.where (noise ==1,255,0)
        noise = np.repeat(noise[:, :, np.newaxis], channel_nb, axis=2)
    if ( noise_fct =="color") :
        noise = generator.integers(0, 256, roi.shape, dtype=np.int64)
    if (noise_fct == "gaussian") :
        noise= generator.normal((((128, ) * channel_nb)),((20, ) * channel_nb), size=roi.shape)
        noise=(noise).astype(np.uint8)
    if (noise_fct =="gaussian_b_w") :        
        noise= generator.normal(128,20, size=roi.shape)
        noise=(noise).astype(np.uint8)
        noise = np.repeat(noise[:, :, np.newaxis], channel_nb, axis=2)
        noise = noise.reshape(roi.shape)

    # Replace the ROI with random noise
    
    image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] = noise

    return Image.fromarray(image)

def replace_plastics_with_noise1(image,bbox,mask, noise_fct):
		# Iterate over the bounding boxes

    image = np.array(image)

    roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 

    if mask : 
        channel_nb = 4
    else : 
        channel_nb =3
        
    if (noise_fct == "b_w") :
        noise = np.random.randint(0, 2, size=[roi.shape[0],roi.shape[1]])
        noise = np.where (noise ==1,255,0)
        noise = np.repeat(noise[:, :, np.newaxis], channel_nb, axis=2)
    if ( noise_fct =="color") :
        noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
    if (noise_fct == "gaussian") :
        noise=np.zeros(roi.shape,dtype=np.uint8)
        cv2.randn(noise,(((128, ) * channel_nb)),((20, ) * channel_nb))
        noise=(noise).astype(np.uint8)
    if (noise_fct =="gaussian_b_w") :
        noise=np.zeros([roi.shape[0],roi.shape[1]],dtype=np.uint8)
        cv2.randn(noise,128,20)
        noise=(noise).astype(np.uint8)
        noise = np.repeat(noise[:, :, np.newaxis], channel_nb, axis=2)
        noise = noise.reshape(roi.shape)

    # Replace the ROI with random noise
    image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] = noise

    return Image.fromarray(image)


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

        self.dir_A = os.path.join(opt.dataroot, 'images', opt.phase) #phase is train/test/...
        self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)

        self.A_paths, self.bbox_paths = make_dataset(self.dir_A, self.dir_bbox)
        self.A_paths = sorted(self.A_paths)
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
        #load data
        A_path = self.A_paths[index]
        bbox_path = self.bbox_paths[index]

        #compute offset
        l = self.opt.loadSize
        f = self.opt.fineSize
        w_offset = random.randint(0, max(0, l - f - 1))
        h_offset = random.randint(0, max(0, l - f - 1))

        bbox = json.load(open(bbox_path))
        A = Image.open(A_path)
        
        #resize
        size_x = A.size[0]
        size_y = A.size[1]
        bbox = {"y":int(bbox["y"]*f/size_y),"x" : int(bbox["x"]*f/size_x), "w" : int(bbox["w"]*f/size_x), "h" : int(bbox["h"]*f/size_y)}
        bbox_x = max(int((bbox['x']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_y = max(int((bbox['y']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_w = max(int((bbox['w']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_h = max(int((bbox['h']/self.opt.fineSize)*self.opt.loadSize), 0)

        if not(self.opt.isTrain) or bbox_y <= h_offset or bbox_x <= w_offset or bbox_h <= h_offset or bbox_w <= w_offset or bbox_y >= h_offset + size_y or bbox_x >= w_offset +size_x/2 or bbox_h >= h_offset +size_y or bbox_w >= w_offset + size_x/2  :
            A = A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
            generator = np.random.default_rng(seed=index)
            B = replace_plastics_with_noise(generator,A,bbox,self.opt.mask,"b_w")

            A = self.transform(A)
            B = self.transform(B)
            A = A[:, :self.opt.fineSize,:self.opt.fineSize]
            B = B[:, :self.opt.fineSize,:self.opt.fineSize]

            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]

        else:

            if self.opt.mask :
                img = Image.fromarray(np.array(A)[:,:,0:3])
                mask = Image.fromarray(np.array(A)[:,:,3])
                img = img.resize((self.opt.loadSize , self.opt.loadSize), Image.BICUBIC)
                mask = mask.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
                A = Image.fromarray(np.concatenate((np.array(img), np.array(mask)[..., np.newaxis]), axis=2))

            else :
                A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)      

            bbox = {"y":bbox_y,"x" : bbox_x, "w" : bbox_w, "h" : bbox_h}
            generator = np.random.default_rng(seed=index)
            B = replace_plastics_with_noise(generator,A,bbox,self.opt.mask,"b_w")
            A = self.transform(A)
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
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
        # print(bbox)
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def name(self):
        return 'NewDataset'
