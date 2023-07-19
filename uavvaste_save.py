# (c) EPFL - LTS5, 2023
import random
import shutil
import os
import sys
import json
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

sys.path.append(os.getcwd())

from coco_api import COCOAnnotations, COCOHandler, PlotType

def image_crop (coco_handler, img_name_list,img_root_dir, image_path_save, bbox_path_save, size ) :
	for img_name in img_name_list : 
		
		image = coco_handler.load_image(img_root_dir, img_name = img_name)
		for annot in coco_handler.get_img_annots(img_name=img_name):
			width, height = image.size
			img = image
			bbox = annot.bbox
			bbox = {"y": bbox[1], "x": bbox[0], "w": bbox[0] + bbox[2], "h": bbox[1] + bbox[3]}
			size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]

			while size_bbox[0] > size or size_bbox[1] > size :
				#print("image_too_large")
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

def generate_random_coordinate(width, height, bbox_list):
	i = 0
	while True :
		new_x = random.randint(0,width-256)
		new_y = random.randint(0,height-256)
		i +=1
		waste_background = False
		for bbox in bbox_list:
			if bbox[0] <=new_x<= bbox[0]+bbox[2] or bbox[0]<=new_x + 256 <=bbox[0]+bbox[2] or bbox[1]<=new_y<=bbox[1]+bbox[3] or bbox[1]<=new_y + 256 <= bbox[1] +bbox[3] or (bbox[0]>=new_x and new_x+256 >=bbox[0]+bbox[2]) or (bbox[1]>=new_y and new_y+256 >=bbox[1]+bbox[3]) :
				waste_background = True
				break

		if not waste_background :
            # Coordinate is not within any of the rectangles
			return new_x, new_y
		if i >=10 :
			print("found nothing")
			return None

def select_empty_background (coco_handler, img_name_list,img_root_dir, image_path_save, bbox_path_save, size ) :
	for img_name in img_name_list : 
		image = coco_handler.load_image(img_root_dir, img_name = img_name)
		width, height = image.size

		bbox_list = [annot.bbox for annot in coco_handler.get_img_annots(img_name=img_name) ]
		random_coordinate = generate_random_coordinate(width, height, bbox_list)
		print("Random coordinate (x, y):", random_coordinate)
		if random_coordinate == None :
			continue

		left = random_coordinate[1]
		top = random_coordinate[0] + size
		right = random_coordinate[1] +size
		bottom = random_coordinate[0]
		img = image.crop((left,bottom,right,top))

		w = random.randint(35,200)
		h = random.randint(35,200)
		x = random.randint(0,size-w-1)
		y = random.randint(0,size-h-1)
		
		bbox = {"y": y, "x": x , "w": x + w , "h": y +h}
		print(bbox)

		im_path = image_path_save + img_name + ".jpg"
		bbox_path = bbox_path_save + img_name + ".json"
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

			if size_bbox[0]<31 or size_bbox[1]<31 : # w:25 et h:70 in the paper
			#if size_bbox[0]<26 or size_bbox[1]<26 or size_bbox[0] in [47,46,39,38,33,32,31,30] or size_bbox[1] in [47,46,39,38,33,32,31,30] :
				# 39/38 : 13 : 2/4 ok
				# 37/36 : 12 : 0/3 ok
				# 35/34 : 11 : 1/3 ok
				# 33/32 : 10 : 1/3 ok
				# 31/30 : 9 : 2/3 non
				# 29/28 : 8 : 0/2 ok
				# 27/26 : 7 : 1/2 ok
				# 25/24 : 6 : 1/2 ok
				i +=1
				os.remove(bbox_path+ os.path.splitext(fname)[0] + ".json")
				os.remove(image_path+ fname)
			print(i)

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

def split_train_test (path_images, path_bbox, destination_images_train,destination_images_test,destination_bbox_train,destination_bbox_test, camera_names) :
	# Create the train and test set directories if they don't exist
	os.makedirs(destination_images_train, exist_ok=True)
	os.makedirs(destination_images_test, exist_ok=True)
	os.makedirs(destination_bbox_train, exist_ok=True)
	os.makedirs(destination_bbox_test, exist_ok=True)

	# Shuffle the camera names to ensure randomness in the train and test sets
	random.seed(3)  # Set a random seed for reproducibility
	shuffled_camera_names = random.sample(camera_names, len(camera_names))

	# Determine the number of cameras to include in the train set and the test set
	train_proportion = 0.8  # 80% of cameras in the train set
	num_cameras_train = int(len(shuffled_camera_names) * train_proportion)
	camera_train = shuffled_camera_names[0:num_cameras_train]
	camera_test = shuffled_camera_names[num_cameras_train:]

	# Iterate over the camera names and copy the corresponding images to the train or test set directories
	i =0
	for fname in os.listdir(path_images):
		#print(fname)
		if any(s in fname for s in camera_train):
			i +=1
			shutil.move(path_images + fname, destination_images_train)
			shutil.move(path_bbox + os.path.splitext(fname)[0] + ".json", destination_bbox_train)
		elif any(s in fname for s in camera_test):
			shutil.move(path_images + fname, destination_images_test)
			shutil.move(path_bbox + os.path.splitext(fname)[0] + ".json", destination_bbox_test)



def main() :
	# os.chdir(os.path.join(os.path.dirname(__file__), "./UAVVaste"))
	# coco_handler = COCOHandler[COCOAnnotations]("./annotations/annotations.json")
	# img_crop_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images_crop/"
	# bbox_crop_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/bbox/"
	# img_noise_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images_noise/"
	
	# image_crop (coco_handler, 
	#        	coco_handler.img_name_list, 
	#   		"/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste/images",
	#   		img_crop_path,
	#   		bbox_crop_path,
	#   		256)
	
	# select_big_plastic(img_crop_path, bbox_crop_path, [256,256],[256,256])
	# replace_plastics_with_noise( img_crop_path, bbox_crop_path, img_noise_path, "b_w")
	
	os.chdir("/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste")
	coco_handler = COCOHandler[COCOAnnotations]("./annotations/annotations.json")
	img_empty_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_empty/images_crop/"
	bbox_empty_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_empty/bbox/"
	noise_empty_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_empty/images_noise/"

	# select_empty_background (coco_handler, 
	# 		coco_handler.img_name_list, 
	# 		"/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste/images",
	# 		img_empty_path,
	# 		bbox_empty_path,
	# 		256)

	replace_plastics_with_noise( img_empty_path, bbox_empty_path, noise_empty_path, "b_w")

	# path_images = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images/"
	# path_bbox = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/bbox/"
	# destination_images_train = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images/train/"
	# destination_images_test = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/images/test/"
	# destination_bbox_train = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/bbox/train/"
	# destination_bbox_test = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/bbox/test/"
	# # list of unique camera names from the dataset
	# camera_names = ["batch_01","batch_02","batch_03","batch_04","batch_05","BATCH_d06","BATCH_d07","BATCH_d08","batch_s01","batch_s02","BATCH_s03","BATCH_s04","BATCH_s05","camera","DJI","GOPR","photo"]
	# split_train_test (path_images, path_bbox, destination_images_train,destination_images_test,destination_bbox_train,destination_bbox_test, camera_names)


if __name__ == "__main__":
	main()
