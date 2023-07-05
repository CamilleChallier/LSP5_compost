import json
import os
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
import tarfile
import pickle
import numpy as np
from tqdm import tqdm
import cv2

path = "tsinghuaDaimlerDataset"

def replace_plastics_with_noise(path_images, path_bounding_boxes, path_save):
    # Iterate over the bounding boxes
    for _, _, fnames in sorted(os.walk(path_images)): 
        for fname in tqdm(fnames[0:1000]) :
        
            image = cv2.imread(path_images + fname,1)

            with open(path_bounding_boxes + os.path.splitext(fname)[0] + ".json" ) as json_file:
                bbox = json.load(json_file)
            
            roi = image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] 

            # Generate random noise
            noise = np.random.randint(0, 256, roi.shape, dtype=np.uint8)

            # Replace the ROI with random noise
            image[bbox["y"]:bbox["h"], bbox["x"]:bbox["w"]] = noise

            plt.imshow(image)

            im_path = path_save + fname
            cv2.imwrite(im_path, image)

replace_plastics_with_noise( path + "/images_full/train/", path + "/bbox2/", path + "/images_noise/train/")

