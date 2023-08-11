# (c) EPFL - LTS5, 2023
import random
import shutil
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

sys.path.append(os.getcwd())

from coco_api import COCOAnnotations, COCOHandler, PlotType

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


class Data_Processing():
	'''Data loader and pre-processing steps for images data'''
	def __init__(self, save_path,size_images=256, with_mask = False ) -> None:
		self.full_images_path = save_path + "images/"
		self.bbox_path = save_path + "bbox/"
		self.location_path = save_path + "location/"
		self.generated_data = save_path +"generated_data/"
		self.size_images = size_images
		self.with_mask = with_mask
	
	def get_full_images_path(self) :
		return self.full_images_path

	def get_bbox_path(self) :
		return self.bbox_path
	
	def get_location_path(self) :
		return self.location_path

	def create_folders(self) :
		os.makedirs(self.get_bbox_path(), exist_ok=True)
		os.makedirs(self.get_full_images_path(), exist_ok=True)
		os.makedirs(self.get_location_path(), exist_ok=True)
	
	def image_crop (self, handler, root_dir) :
		for img_name in handler.img_name_list: 
			
			image = handler.load_image(root_dir + "images", img_name = img_name)

			if self.with_mask :

				mask = Image.open(os.path.join(root_dir+"masks/", img_name +".png"))
				arr_img1 = np.array(image)
				arr_img2 = np.array(mask.convert('L'))
				arr_img2_resized = arr_img2[..., np.newaxis]  # Add a new axis at the end to make it (256, 256, 1)
				final_image = Image.fromarray(np.concatenate((arr_img1, arr_img2_resized), axis=2))
			else :
				final_image = image

			for annot in handler.get_img_annots(img_name=img_name):
				width, height = final_image.size
				img = final_image
				bbox = annot.bbox
				bbox = {"y": bbox[1], "x": bbox[0], "w": bbox[0] + bbox[2], "h": bbox[1] + bbox[3]}
				size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]
				factor = 1

				while size_bbox[0] > self.size_images or size_bbox[1] > self.size_images :
					img = img.resize((int(width/2),int(height/2)), Image.BICUBIC)
					factor *=2
					bbox = {"y": int(bbox["y"]/2), "x": int(bbox["x"]/2), "w": int(bbox["w"]/2), "h": int(bbox["h"]/2)}
					size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]
					width, height = img.size

				left = bbox["x"]+int(int(size_bbox[1]/2)-self.size_images/2)
				top = bbox["y"]+int(int(size_bbox[0]/2)+self.size_images/2)
				right = bbox["x"]+int(int(size_bbox[1]/2)+self.size_images/2)
				bottom = bbox["y"]+int(int(size_bbox[0]/2)-self.size_images/2)

				if left <0 :
					#print("1")
					diff = 0-left
					left = 0
					right = right + diff
				if right > width :
					#print("2")
					diff = right -width
					right = width
					left = left - diff
				if top > height :
					#print("3")
					diff = top -height
					top = height
					bottom = bottom - diff
				if bottom <0 :
					#print("4")
					diff = 0 - bottom
					bottom = 0
					top = top + diff

				bbox = {"y": bbox["y"]-bottom, "x": bbox["x"]-left, "w": bbox["w"]-left, "h": bbox["h"]-bottom}

				img = img.crop((left,bottom,right,top))
				
				location = {"left": left*factor, "bottom":bottom*factor, "right": right*factor, "top": top*factor}

				im_path = self.full_images_path + img_name + "_annot" + str(annot.id) + ".png"
				bbox_path = self.bbox_path + img_name + "_annot" + str(annot.id) + ".json"
				location_path = self.location_path + img_name + "_annot" + str(annot.id) + ".json"

				img.save(im_path)
				with open(location_path, "w") as outfile:
					json.dump(location, outfile)

				with open(bbox_path, "w") as outfile:
					json.dump(bbox, outfile)

	
	def select_empty_background (self, handler,root_dir) :

		for img_name in handler.img_name_list : 
			image = handler.load_image(root_dir + "images", img_name = img_name)

			if self.with_mask :
				mask = Image.open(os.path.join(root_dir+"masks/", img_name +".png"))
				arr_img1 = np.array(image)
				arr_img2 = np.array(mask.convert('L'))
				arr_img2_resized = arr_img2[..., np.newaxis]  # Add a new axis at the end to make it (256, 256, 1)
				final_image = Image.fromarray(np.concatenate((arr_img1, arr_img2_resized), axis=2))
			else :
				final_image = image

			width, height = final_image.size

			bbox_list = [annot.bbox for annot in handler.get_img_annots(img_name=img_name) ]
			random_coordinate = generate_random_coordinate(width, height, bbox_list)
			print("Random coordinate (x, y):", random_coordinate)
			if random_coordinate == None :
				continue

			left = random_coordinate[0]
			top = random_coordinate[1] + self.size_images
			right = random_coordinate[0] +self.size_images
			bottom = random_coordinate[1]
			img = final_image.crop((left,bottom,right,top))

			location = {"left": left, "bottom":bottom, "right": right, "top": top}
			w = random.randint(35,200)
			h = random.randint(35,200)
			x = random.randint(0,self.size_images-w-1)
			y = random.randint(0,self.size_images-h-1)
			
			bbox = {"y": y, "x": x , "w": x + w , "h": y +h}

			os.makedirs(self.full_images_path + "test/", exist_ok=True)
			os.makedirs(self.bbox_path + "test/", exist_ok=True)
			im_path = self.full_images_path + "test/"+ img_name + ".png"
			bbox_path = self.bbox_path + "test/" + img_name + ".json"
			location_path = self.location_path + img_name + ".json"

			img.save(im_path)
			with open(location_path, "w") as outfile:
				json.dump(location, outfile)

			with open(bbox_path, "w") as outfile:
				json.dump(bbox, outfile)

	def select_big_plastic (self,min_size=32):
		#select only plastics large enought
		i = 0
		for _, _, fnames in sorted(os.walk(self.full_images_path)): #for each folder
			for fname in fnames:

				with open(self.bbox_path+ os.path.splitext(fname)[0] + ".json" ) as json_file:
					bbox = json.load(json_file)

				size_bbox = [int(bbox["h"])-int(bbox["y"]), int(bbox["w"])-int(bbox["x"])] 

				if size_bbox[0]<min_size or size_bbox[1]<min_size :
					i +=1
					os.remove(self.bbox_path+ os.path.splitext(fname)[0] + ".json")
					os.remove(self.full_images_path+ fname)
					os.remove(self.location_path + os.path.splitext(fname)[0] + ".json")
					print(i)

	def replace_plastics_with_noise(self, noise_fct):
		# Iterate over the bounding boxes

		for _, _, fnames in sorted(os.walk(self.full_images_path)): 
			for fname in tqdm(fnames) :
			
				image = np.array(Image.open(os.path.join(self.full_images_path + fname)))

				with open(self.bbox_path + os.path.splitext(fname)[0] + ".json" ) as json_file:
					bbox = json.load(json_file)

				roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 

				if self.with_mask : 
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

				im_path = self.noise_path + fname
				Image.fromarray(image).save(im_path)

	def split_train_test (self, camera_names) :
		# Create the train and test set directories if they don't exist

		destination_images_train = self.full_images_path + "/train/"
		destination_images_test = self.full_images_path + "test/"
		destination_bbox_train = self.bbox_path +"train/"
		destination_bbox_test =  self.bbox_path +"test/"

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
		for fname in os.listdir(self.full_images_path):
			#print(fname)
			if any(s in fname for s in camera_train):
				i +=1
				shutil.move(self.full_images_path + fname, destination_images_train)
				shutil.move(self.bbox_path + os.path.splitext(fname)[0] + ".json", destination_bbox_train)
			elif any(s in fname for s in camera_test):
				shutil.move(self.full_images_path + fname, destination_images_test)
				shutil.move(self.bbox_path + os.path.splitext(fname)[0] + ".json", destination_bbox_test)

	def combine_images (self) :
		for fname in os.listdir(self.full_images_path):
			path_A = self.full_images_path +  fname
			path_B = self.noise_path +  fname
			if os.path.isfile(path_A) and os.path.isfile(path_B):
				
				im_A = np.array(Image.open(os.path.join(path_A)))
				im_B = np.array(Image.open(os.path.join(path_B)))
				im_AB = np.concatenate([im_A, im_B], 1)
				
				Image.fromarray(im_AB).save(self.image_aligned_path + fname)

	def paste_generated_data (self, path_result, root_dir) :
		os.makedirs(self.generated_data, exist_ok=True)

		for fname in os.listdir(path_result):
			if '_'.join(fname.split('_')[-2:]) == "fake_B.png" : 
				img_name = fname.rsplit('_', 2)[0]
				if img_name.split('_')[-1][0:5] == "annot" :
					big_img_name = img_name.rsplit('_', 1)[0]
				else :
					big_img_name = img_name

				if (big_img_name[0]== "G" or big_img_name[0]=="D") :
					big_img_name = big_img_name +".JPG"
				else :
					big_img_name = big_img_name +".jpg"

				image = np.array(Image.open(os.path.join(root_dir +"/images", big_img_name)))

				result = np.array(Image.open(path_result + "/"+ fname))

				with open(self.location_path +img_name + ".json" ) as json_file:
					crop = json.load(json_file)

				image[crop["bottom"]:crop["top"],crop["left"]:crop["right"], :] = result
				Image.fromarray(image).save(self.generated_data + img_name + ".png")


			if '_'.join(fname.split('_')[-3:]) == "fake_B_mask.png" : 
				mask_name = fname.rsplit('_', 3)[0]

				if mask_name.split('_')[-1][0:5] == "annot" :
					big_mask_name = mask_name.rsplit('_', 1)[0]
				else :
					big_mask_name = mask_name

				mask = np.array(Image.open(os.path.join(root_dir +"masks", big_mask_name +".png")))

				print(path_result + "/"+ fname)
				result = np.array(Image.open(path_result + "/"+ fname))

				with open(self.location_path + mask_name + ".json" ) as json_file:
					crop = json.load(json_file)

				mask[crop["bottom"]:crop["top"],crop["left"]:crop["right"]] = result
				Image.fromarray(mask).save(self.generated_data + mask_name + "_mask.png")


def main() :
	np.set_printoptions(threshold=sys.maxsize)

	#create data handler from coco dataset
	os.chdir(os.path.join(os.path.dirname(__file__), "./UAVVaste"))
	print(os.path.join(os.path.dirname(__file__), "./UAVVaste"))
	coco_handler = COCOHandler[COCOAnnotations]("./annotations/annotations.json")
	data_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste/"
	camera_names = ["batch_01","batch_02","batch_03","batch_04","batch_05","BATCH_d06","BATCH_d07","BATCH_d08","batch_s01","batch_s02","BATCH_s03","BATCH_s04","BATCH_s05","camera","DJI","GOPR","photo"]


	save_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_mask/"
	data = Data_Processing(save_path,size_images=256, with_mask = True )
	data.create_folders()
	
	#Part 1
	# create the training/testing dataset : crop images around plastics
	data.image_crop (coco_handler, data_path)
	data.select_big_plastic(min_size = 32)
	data.split_train_test (camera_names)

	# Part 2
	# create a test set with just background without plastics
	# save_path_wp = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_mask_empty/"
	# data_test_wp = Data_Processing(save_path_wp,size_images=256,  with_mask = False  )
	# data_test_wp.create_folders()
	# data_test_wp.select_empty_background (coco_handler, data_path)

	# Part 3
	# path_result = "/home/ccamille/biowaste_GAN/LSP5_compost/results/biowaste_mn_all_TG1/test_latest/images"
	# data.paste_generated_data(path_result,data_path)

if __name__ == "__main__":
	main()
