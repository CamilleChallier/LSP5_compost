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
	def __init__(self, save_path,size_images=256 ) -> None:
		self.full_images_path = save_path + "images_crop/"
		self.bbox_path = save_path + "bbox/"
		self.location_path = save_path + "location/"
		self.noise_path = save_path + "images_noise/"
		self.image_aligned_path = save_path + "images/"
		self.generated_data = save_path +"generated_data/"
		self.size_images = size_images
	
	def get_full_images_path(self) :
		return self.full_images_path
    
	def get_noise_path(self) :
		return self.noise_path
    
	def get_image_aligned_path(self) :
		return self.image_aligned_path

	def get_bbox_path(self) :
		return self.bbox_path
	
	def get_location_path(self) :
		return self.location_path
	
	def create_folders(self) :
		os.makedirs(self.get_bbox_path(), exist_ok=True)
		os.makedirs(self.get_noise_path(), exist_ok=True)
		os.makedirs(self.get_full_images_path(), exist_ok=True)
		os.makedirs(self.get_image_aligned_path(), exist_ok=True)
		os.makedirs(self.get_location_path(), exist_ok=True)

	def image_crop (self, handler, img_root_dir) :
		for img_name in handler.img_name_list: 
			
			image = handler.load_image(img_root_dir, img_name = img_name)
			for annot in handler.get_img_annots(img_name=img_name):
				width, height = image.size
				img = image
				bbox = annot.bbox
				bbox = {"y": bbox[1], "x": bbox[0], "w": bbox[0] + bbox[2], "h": bbox[1] + bbox[3]}
				size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]

				while size_bbox[0] > self.size_images or size_bbox[1] > self.size_images :
					#print("image_too_large")
					img = img.resize((int(width/2),int(height/2)), Image.BICUBIC)
					bbox = {"y": int(bbox["y"]/2), "x": int(bbox["x"]/2), "w": int(bbox["w"]/2), "h": int(bbox["h"]/2)}
					size_bbox = [bbox["h"]-bbox["y"], bbox["w"]-bbox["x"]]
					width, height = img.size
				#print(size_bbox)
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

				#print(left-right,top-bottom)
				bbox = {"y": bbox["y"]-bottom, "x": bbox["x"]-left, "w": bbox["w"]-left, "h": bbox["h"]-bottom}

				img = img.crop((left,bottom,right,top))

				location = {"left": left, "bottom":bottom, "right": right, "top": top}

				im_path = self.full_images_path + img_name + "_annot" + str(annot.id) + ".jpg"
				bbox_path = self.bbox_path + img_name + "_annot" + str(annot.id) + ".json"
				location_path = self.location_path + img_name + "_annot" + str(annot.id) + ".json"
				img.save(im_path)
				with open(location_path, "w") as outfile:
					json.dump(location, outfile)

				with open(bbox_path, "w") as outfile:
					json.dump(bbox, outfile)

	
	def select_empty_background (self, handler,img_root_dir) :

		for img_name in handler.img_name_list : 
			image = handler.load_image(img_root_dir, img_name = img_name)
			width, height = image.size

			bbox_list = [annot.bbox for annot in handler.get_img_annots(img_name=img_name) ]
			random_coordinate = generate_random_coordinate(width, height, bbox_list)
			print("Random coordinate (x, y):", random_coordinate)
			if random_coordinate == None :
				continue

			left = random_coordinate[0]
			top = random_coordinate[1] + self.size_images
			right = random_coordinate[0] +self.size_images
			bottom = random_coordinate[1]
			img = image.crop((left,bottom,right,top))

			location = {"left": left, "bottom":bottom, "right": right, "top": top}
			w = random.randint(35,200)
			h = random.randint(35,200)
			x = random.randint(0,self.size_images-w-1)
			y = random.randint(0,self.size_images-h-1)
			
			bbox = {"y": y, "x": x , "w": x + w , "h": y +h}

			im_path = self.full_images_path + img_name + ".jpg"
			bbox_path = self.bbox_path + img_name + ".json"
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

				if size_bbox[0]<min_size or size_bbox[1]<min_size : # w:25 et h:70 in the paper
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
					os.remove(self.bbox_path+ os.path.splitext(fname)[0] + ".json")
					os.remove(self.full_images_path+ fname)
					os.remove(self.location_path + os.path.splitext(fname)[0] + ".json")
					print(i)

	def replace_plastics_with_noise(self, noise_fct):
		# Iterate over the bounding boxes

		for _, _, fnames in sorted(os.walk(self.full_images_path)): 
			for fname in tqdm(fnames) :
			
				image = cv2.imread(self.full_images_path + fname,1)

				with open(self.bbox_path + os.path.splitext(fname)[0] + ".json" ) as json_file:
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

				im_path = self.noise_path + fname
				cv2.imwrite(im_path, image)

	def split_train_test (self, camera_names) :
		# Create the train and test set directories if they don't exist

		destination_images_train = self.image_aligned_path + "/train/"
		destination_images_test = self.image_aligned_path + "test/"
		destination_bbox_train = self.bbox_path +"train/"
		destination_bbox_test =  self.bbox_path +" test/"

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
		for fname in os.listdir(self.image_aligned_path):
			#print(fname)
			if any(s in fname for s in camera_train):
				i +=1
				shutil.move(self.image_aligned_path + fname, destination_images_train)
				shutil.move(self.bbox_path + os.path.splitext(fname)[0] + ".json", destination_bbox_train)
			elif any(s in fname for s in camera_test):
				shutil.move(self.image_aligned_path + fname, destination_images_test)
				shutil.move(self.bbox_path + os.path.splitext(fname)[0] + ".json", destination_bbox_test)

	def combine_images (self) :
		for fname in os.listdir(self.full_images_path):
			path_A = self.full_images_path +  fname
			path_B = self.noise_path +  fname
			if os.path.isfile(path_A) and os.path.isfile(path_B):
				im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
				im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
				im_AB = np.concatenate([im_A, im_B], 1)
				cv2.imwrite(self.image_aligned_path + fname, im_AB)

	def paste_generated_data (self, path_result,handler, img_root_dir) :
		os.makedirs(self.generated_data, exist_ok=True)

		for fname in os.listdir(path_result)[30:40]:
			if fname.split('_')[-1] == "display.png" : 
				img_name = fname.rsplit('_', 1)[0]
				print(img_name)
				if img_name.split('_')[-1][0:5] == "annot" :
					big_img_name = img_name.rsplit('_', 1)[0]
				else :
					big_img_name = img_name

				if (big_img_name[0]== "G" or big_img_name[0]=="D") :
					big_img_name = big_img_name +".JPG"
				else :
					big_img_name = big_img_name +".jpg"

				print(os.path.join(img_root_dir, big_img_name))
				image = cv2.imread(os.path.join(img_root_dir, big_img_name),1)
				plt.imshow(image)
				plt.show()
				print(image.shape)
				print(big_img_name)

				result = cv2.imread(path_result + "/"+ fname,1)
				print(result.shape)
				plt.imshow(result)
				plt.show()

				ret = cv2.imread("/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_wb/images_crop/BATCH_d06_img_1750.jpg",1)
				print(ret.shape)
				plt.imshow(ret)
				plt.show()

				print(self.location_path +img_name + ".json" )
				with open(self.location_path +img_name + ".json" ) as json_file:
					crop = json.load(json_file)
				print(crop)
				image[crop["bottom"]:crop["top"],crop["left"]:crop["right"], :] = result
				#cv2.imwrite(self.generated_data, image)
				plt.imshow(image)
				plt.show()



def main() :

	#create data handler from coco dataset
	os.chdir(os.path.join(os.path.dirname(__file__), "./UAVVaste"))
	coco_handler = COCOHandler[COCOAnnotations]("./annotations/annotations.json")
	data_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste/images"
	camera_names = ["batch_01","batch_02","batch_03","batch_04","batch_05","BATCH_d06","BATCH_d07","BATCH_d08","batch_s01","batch_s02","BATCH_s03","BATCH_s04","BATCH_s05","camera","DJI","GOPR","photo"]


	save_path = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data/"
	data = Data_Processing(save_path,size_images=256 )
	data.create_folders()
	
	#crop images around plastics
	data.image_crop (coco_handler, data_path)
	data.select_big_plastic(min_size = 32)
	#create noisy data
	data.replace_plastics_with_noise("b_w")
	os.makedirs(data.get_image_aligned_path(), exist_ok=True)
	data.combine_images ()
	data.split_train_test (camera_names)


	# #create a test set with just background without plastics
	save_path_wp = "/home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_wb/"
	data_test_wp = Data_Processing(save_path_wp,size_images=256 )
	data_test_wp.create_folders()
	data_test_wp.select_empty_background (coco_handler, data_path)
	data_test_wp.replace_plastics_with_noise("b_w")
	data_test_wp.combine_images ()

	path_result = "/home/ccamille/biowaste_GAN/LSP5_compost/results/biowaste_128_TG1/test_latest/images"
	data_test_wp.paste_generated_data(path_result, coco_handler,data_path)



if __name__ == "__main__":
	main()
