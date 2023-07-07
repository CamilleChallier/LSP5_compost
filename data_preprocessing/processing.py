import json
import os
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import tarfile
import pickle
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

path = "tsinghuaDaimlerDataset"

def file_extraction(data_path, save_path) :
    """
    Extract tar.gz files
    """
    #extract images or labels
    # open file
    file = tarfile.open(data_path)
    # extracting file
    file.extractall(save_path)
    file.close()

def create_bbox(path) : 
    """
    create bbox from label
    """
    for _, _, fnames in sorted(os.walk(path+"/label/labelData/train/tsinghuaDaimlerDataset/")): #for each folder
        for fname in fnames[1500:2000] :
            with open("tsinghuaDaimlerDataset/label/labelData/train/tsinghuaDaimlerDataset/"+fname) as json_file:
                im = json.load(json_file)
            bbox = {"y": im["children"][0]["minrow"], "x": im["children"][0]["mincol"], "w": (im["children"][0]["maxcol"]), "h": im["children"][0]["maxrow"]}
            with open(path +"/images_aligned/bbox/test/"+ os.path.splitext(im["imagename"])[0]+".json", "w") as outfile:
                json.dump(bbox, outfile)

#crop image and modify bbox
def image_crop (image_path, bbox_path, image_path_save, bbox_path_save ) :
    for _, _, fnames in sorted(os.walk(image_path)): #for each folder
        for fname in fnames :
            AB = Image.open(image_path+ fname).convert('RGB')
            #AB = cv2.imread("tsinghuaDaimlerDataset/images_full/train/"+fname,1)
            width, height = AB.size
            left = int((width - height) / 2)
            right = left + height
            im1 = AB.crop((left, 0, right, height))

            im1.save(image_path_save+ fname)

            with open(bbox_path+ os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)

            bbox2 = {"y": bbox["y"], "x": bbox["x"]-left, "w": bbox["w"]-left, "h": bbox["h"]}
            if bbox2["x"]>= 1024 and bbox2["w"] >= 1024 :
                bbox2["x"] = 1024
                bbox2["w"] = 1024 
            if bbox2["x"]<= 1024 and bbox2["w"] > 1024 :
                bbox2["w"] = 1024 
            if bbox2["x"] <0 :
                bbox2["x"] =0
            if bbox2["w"] <0 :
                bbox2["w"] =0
            with open(bbox_path_save+ os.path.splitext(fname)[0] +".json", "w") as outfile:
                json.dump(bbox2, outfile)

def replace_plastics_with_noise(path_images, path_bounding_boxes, path_save, noise = "b_w"):
    # Iterate over the bounding boxes
    for _, _, fnames in sorted(os.walk(path_images)): 
        for fname in tqdm(fname) :
        
            image = cv2.imread(path_images + fname,1)

            with open(path_bounding_boxes + os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)
            
            roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 

            if (noise == "b_w") :
                noise = np.random.randint(0, 2, size=[roi.shape[0],roi.shape[1]])
                noise = np.where (noise ==1,255,0)
                noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
            if ( noise =="color") :
                noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
            if (noise == "gaussian") :
                mean = 0
                var = 0.1
                sigma = var**0.5
                noise = np.random.normal(mean,sigma,roi.shape)
                noise = noise.reshape(roi.shape)*255

            # Replace the ROI with random noise
            image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] = noise

            plt.imshow(image)

            im_path = path_save + fname
            cv2.imwrite(im_path, image)

def select_big_plastic (image_path,bbox_path):
    #select only plastics large enought
    for _, _, fnames in sorted(os.walk(image_path)): #for each folder
        for fname in fnames[0:1000]:

            with open(bbox_path+ os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)
            h = 512
            w_total = 512*2
            size_y = 1024
            size_x = 1024 *2
            size_bbox = [int(bbox["h"]*h/size_y)-int(bbox["y"]*h/size_y), int(bbox["w"]*w_total/size_x)-int(bbox["x"]*w_total/size_x)] 

            if size_bbox[0]<48 or size_bbox[1]<48 :
                print(size_bbox)
                os.remove(bbox_path+ os.path.splitext(fname)[0] + ".json")
                os.remove(image_path+ fname)

file_extraction('tsinghuaDaimlerDataset/tdcb_leftImg8bit_train.tar.gz','tsinghuaDaimlerDataset/images_full/test/')
create_bbox(path)
image_crop (path + "/images_full/test/","tsinghuaDaimlerDataset/images_aligned/bbox/test/", "tsinghuaDaimlerDataset/images_full_crop/test/","tsinghuaDaimlerDataset/images_aligned_crop/bbox/test/")
replace_plastics_with_noise( path + "/images_full_crop/test/", path + "/images_aligned_crop/bbox/test/", path + "/images_noise_crop/test/")
#combine
select_big_plastic(path + "/images_aligned_crop/images/test/", "tsinghuaDaimlerDataset/images_aligned_crop/bbox/test/")
