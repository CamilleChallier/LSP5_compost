import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import PIL
from pdb import set_trace as st


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot,"images", opt.phase + 'A') #specific folder trainA
        self.dir_B = os.path.join(opt.dataroot, "images", opt.phase + 'B')
        self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        
        self.A_paths, self.bbox_paths = make_dataset(self.dir_A, self.dir_bbox)
        self.B_paths, self.bbox_paths = make_dataset(self.dir_B, self.dir_bbox)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.bbox_paths = sorted(self.bbox_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.RandomCrop(opt.fineSize),
                        transforms.ToTensor(), # convert image to tensor
                        transforms.Normalize(((0.5, ) * opt.input_nc),
                                               ((0.5, ) * opt.input_nc))] # list that holds the defined transformations. In this case, it contains the ToTensor() transformation and the Normalize() transformation.

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        import numpy as np

        if self.opt.mask :
            A = Image.fromarray(np.array(A_img)[:,:,0:3])
            B = Image.fromarray(np.array(B_img)[:,:,0:3])
            A_mask = Image.fromarray(np.array(A_img)[:,:,3])
            B_mask = Image.fromarray(np.array(B_img)[:,:,3])
            A = A.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            A_mask = A_mask.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            B = B.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            B_mask = B_mask.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)

            A = Image.fromarray(np.concatenate((np.array(A), np.array(A_mask)[..., np.newaxis]), axis=2))
            B = Image.fromarray(np.concatenate((np.array(B), np.array(B_mask)[..., np.newaxis]), axis=2))
        else :
            A = A_img.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            B = B_img.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)

        A = self.transform(A)
        B= self.transform(B)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
