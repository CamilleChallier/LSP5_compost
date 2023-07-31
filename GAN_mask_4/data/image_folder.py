###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir_img, dir_bbox):
    """_summary_

    Args:
        dir_img (str): path of images
        dir_bbox (str): path of bbox

    Returns:
        list : list of all path of images and list of all path of bbox
    """
    images = []
    bbox = []
    assert os.path.isdir(dir_img), '%s is not a valid directory' % dir_img

    for root, _, fnames in sorted(os.walk(dir_img)): #for each folder
        for fname in fnames: #for each files
            if is_image_file(fname): 
                img_path = os.path.join(root, fname)
                images.append(img_path)
                bbox_path = os.path.join(dir_bbox, fname.split('.')[0])+'.json'
                #print(bbox_path)
                bbox.append(bbox_path)

    return images, bbox 

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
