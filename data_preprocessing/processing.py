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

def image_resize(image_path, bbox_path, image_path_save, bbox_path_save, size=286 ) :
    AB = Image.open(image_path).convert('RGB')
    width, height = AB.size
    AB = AB.resize((size, size), Image.BICUBIC)
    with open(bbox_path ) as json_file:
        bbox = json.load(json_file)

    x = int((bbox["x"] * size) /width)
    y = int((bbox["y"] *size) /height)
    w = int(bbox["w"] * size /width)
    h = int(bbox["h"] * size /height)
    bbox2 = {"y": y, "x": x, "w": w, "h": h}

    with open(bbox_path_save, "w") as outfile:
        json.dump(bbox2, outfile)

    AB.save(image_path_save)

def replace_plastics_with_noise(path_images, path_bounding_boxes, path_save, noise_fct):
    # Iterate over the bounding boxes
    for _, _, fnames in sorted(os.walk(path_images)): 
        for fname in tqdm(fnames) :
        
            image = cv2.imread(path_images + fname,1)

            with open(path_bounding_boxes + os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)
            
            roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 

            if (noise_fct == "b_w") :
                noise = np.random.randint(0, 2, size=[roi.shape[0],roi.shape[1]])
                noise = np.where (noise ==1,255,0)
                noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
            if ( noise_fct =="color") :
                noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)
            if (noise_fct == "gaussian") :
                noise=np.zeros(roi.shape,dtype=np.uint8)
                cv2.randn(noise,(128,128,128),(20,20,20))
                noise=(noise).astype(np.uint8)
            if (noise_fct =="gaussian_b_w") :
                noise=np.zeros([roi.shape[0],roi.shape[1]],dtype=np.uint8)
                cv2.randn(noise,128,20)
                noise=(noise).astype(np.uint8)
                noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                noise = noise.reshape(roi.shape)

            # Replace the ROI with random noise
            image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] = noise

            plt.imshow(image)

            im_path = path_save + fname
            cv2.imwrite(im_path, image)

def select_big_plastic (image_path, bbox_path, output_size, input_size):
    #select only plastics large enought
    for _, _, fnames in sorted(os.walk(image_path)): #for each folder
        for fname in fnames:

            with open(bbox_path+ os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)
            h = output_size[0]
            w_total = output_size[1]
            size_y = input_size[0]
            size_x = input_size[1]
            size_bbox = [int(bbox["h"]*h/size_y)-int(bbox["y"]*h/size_y), int(bbox["w"]*w_total/size_x)-int(bbox["x"]*w_total/size_x)] 

            if size_bbox[0]<70 or size_bbox[1]<48 : # w:25 et h:70 avec crop, mais avec 48 ca marche a voir
                os.remove(bbox_path+ os.path.splitext(fname)[0] + ".json")
                os.remove(image_path+ fname)
                print("removed")

def select_square_around_plastic (image_path,bbox_path,image_path_save, bbox_path_save,size) :
    for _, _, fnames in sorted(os.walk(image_path)): #for each folder
        for fname in fnames :
            AB = Image.open(image_path+ fname).convert('RGB')
            width, height = AB.size
            with open(bbox_path+ os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)
            size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]

            while size_bbox[0] > size or size_bbox[1] > size :
                AB = AB.resize((int(width/2),int(height/2)), Image.BICUBIC)
                bbox = {"y": int(bbox["y"]/2), "x": int(bbox["x"]/2), "w": int(bbox["w"]/2), "h": int(bbox["h"]/2)}
                size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]
                width, height = AB.size

            left = bbox["x"]+int(size_bbox[1]/2-size/2)
            top = bbox["y"]+int(size_bbox[0]/2+size/2)
            right = bbox["x"]+int(size_bbox[1]/2+size/2)
            bottom = bbox["y"]+int(size_bbox[0]/2-size/2)

            if left <0 :
                diff = 0-left
                left = 0
                right = right + diff
            if right > width :
                diff = right -width
                right = width
                left = left - diff
            if top > height :
                diff = top -height
                top = height
                bottom = bottom - diff
            if bottom <0 :
                diff = 0 - bottom
                bottom = 0
                top = top + diff

            AB = AB.crop((left,bottom,right,top))

            AB.save(image_path_save+ fname)

            bbox2 = {"y": bbox["y"]-bottom, "x": bbox["x"]-left, "w": bbox["w"]-left, "h": bbox["h"]-bottom}
            
            with open(bbox_path_save+ os.path.splitext(fname)[0] +".json", "w") as outfile:
                json.dump(bbox2, outfile)

#select_square_around_plastic ("tsinghuaDaimlerDataset/images_full/train/","tsinghuaDaimlerDataset/images_aligned/bbox/train/","tsinghuaDaimlerDataset/images_crop_bb/images/train/", "tsinghuaDaimlerDataset/images_crop_bb/bbox/train/",256)

# file_extraction('tsinghuaDaimlerDataset/tdcb_leftImg8bit_train.tar.gz','tsinghuaDaimlerDataset/images_full/test/')
# create_bbox(path)
# image_crop (path + "/images_full/test/","tsinghuaDaimlerDataset/images_aligned/bbox/test/", "tsinghuaDaimlerDataset/images_full_crop/test/","tsinghuaDaimlerDataset/images_aligned_crop/bbox/test/")
#replace_plastics_with_noise( path + "/images_crop_bb/images/train/", path + "/images_crop_bb/bbox/train/", path + "/images_crop_bb_gaussian/images/train/", "gaussian")
# #combine
select_big_plastic(path + "/images_aligned_bb/images/train/", "tsinghuaDaimlerDataset/images_crop_bb/bbox/train/", [256,256],[256,256])

# python GAN/datasets/combine_A_and_B.py --fold_A /app/LSP5_compost/data_preprocessing/tsinghuaDaimlerDataset/images_crop_bb/images  --fold_B /app/LSP5_compost/data_preprocessing/tsinghuaDaimlerDataset/images_crop_bb_gaussian/images --fold_AB /app/LSP5_compost/data_preprocessing/tsinghuaDaimlerDataset/images_aligned_bb/images2