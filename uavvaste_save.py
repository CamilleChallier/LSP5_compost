# (c) EPFL - LTS5, 2023

import os
import sys
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

sys.path.append(os.getcwd())

from coco_api import COCOAnnotations, COCOHandler, PlotType

def image_crop (coco_handler, img_name_list,img_root_dir, image_path_save, bbox_path_save, size ) :
	for img_name in img_name_list : 
		
		image = coco_handler.load_image(img_root_dir, img_name = img_name)
		width, height = image.size
		for annot in coco_handler.get_img_annots(img_name=img_name):
			img = image
			bbox = annot.bbox
			bbox = {"y": bbox[1], "x": bbox[0], "w": bbox[0] + bbox[2], "h": bbox[1] + bbox[3]}
			size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]

			while size_bbox[0] > size or size_bbox[1] > size :
				print("image_too_large")
				img = img.resize((int(width/2),int(height/2)), Image.BICUBIC)
				bbox = {"y": int(bbox["y"]/2), "x": int(bbox["x"]/2), "w": int(bbox["w"]/2), "h": int(bbox["h"]/2)}
				size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]
				width, height = img.size

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

			bbox = {"y": bbox["y"]-bottom, "x": bbox["x"]-left, "w": bbox["w"]-left, "h": bbox["h"]-bottom}

			img = img.crop((left,bottom,right,top))

			im_path = image_path_save + img_name + "_annot" + str(annot.id) + ".jpg"
			bbox_path = bbox_path_save + img_name + "_annot" + str(annot.id) + ".json"
			img.save(im_path)

			with open(bbox_path, "w") as outfile:
				json.dump(bbox, outfile)

def select_big_plastic (image_path, bbox_path, output_size, input_size):
    #select only plastics large enought
	i = 0
	for _, _, fnames in sorted(os.walk(image_path)): #for each folder
		for fname in fnames:

			with open(bbox_path+ os.path.splitext(fname)[0] + ".json" ) as json_file:
				bbox = json.load(json_file)
			h = output_size[0]
			w_total = output_size[1]
			size_y = input_size[0]
			size_x = input_size[1]
			size_bbox = [int(bbox["h"]*h/size_y)-int(bbox["y"]*h/size_y), int(bbox["w"]*w_total/size_x)-int(bbox["x"]*w_total/size_x)] 

			#if size_bbox[0]<48 or size_bbox[1]<48 : # w:25 et h:70 in the paper
			if size_bbox[0]<26 or size_bbox[1]<26 or size_bbox[0] in [47,46,39,38,33,32,31,30] or size_bbox[1] in [47,46,39,38,33,32,31,30] :
				# 48 : 18 : 2/5 ok
				# 47/46 : 17 : 3/5 non
				# 45/44 : 16 : 0/4 ok
				# 43/42 : 15 : 1/4 ok
				# 41/40 : 14 : 2/4 ok
				# 39/38 : 13 : 3/4 non
				# 37/36 : 12 : 0/3 ok
				# 35/34 : 11 : 1/3 ok
				# 33/32 : 10 : 2/3 non
				# 31/30 : 9 : 3/3 non
				# 29/28 : 8 : 0/2 ok
				# 27/26 : 7 : 1/2 ok
				# 25/24 : 6 : 2/2 non
				i +=1
				print(i)
				os.remove(bbox_path+ os.path.splitext(fname)[0] + ".json")
				os.remove(image_path+ fname)

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

            im_path = path_save + fname
            cv2.imwrite(im_path, image)


def save_plot ( ):
	os.chdir(os.path.join(os.path.dirname(__file__), "./UAVVaste"))
	coco_handler = COCOHandler[COCOAnnotations]("./annotations/annotations.json")
	coco_handler.plot(coco_handler.img_name_list, "images", "annot", PlotType.ANNOTATION)
	coco_handler.plot(coco_handler.img_name_list, "images", "masks", PlotType.MASK)

def main() :
	os.chdir(os.path.join(os.path.dirname(__file__), "./UAVVaste"))
	coco_handler = COCOHandler[COCOAnnotations]("./annotations/annotations.json")
	img_crop_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images_crop/"
	bbox_crop_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/bbox/"
	img_noise_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images_noise/"
	image_crop (coco_handler, 
	       	coco_handler.img_name_list, 
	  		"/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste/images",
	  		img_crop_path,
	  		bbox_crop_path,
	  		256)
	
	select_big_plastic(img_crop_path, bbox_crop_path, [256,256],[256,256])
	replace_plastics_with_noise( img_crop_path, bbox_crop_path, img_noise_path, "b_w")


if __name__ == "__main__":
	main()
