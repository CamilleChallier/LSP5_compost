# PS-GAN
This is a Pytorch implementation of [Pedestrian-Synthesis-GAN](https://github.com/yifanjiang19/Pedestrian-Synthesis-GAN) on UAVVaste dataset.

## Introduction

We propose an augmentation method to generate RGB + mask pairs with a GAN.

## Usage

This project use visdom for visualization. Run it to see the training process.

```bash
python -m visdom.server
```

The `UAVVaste` dataset repository is added to the project as a submodule, the images need to be downloaded using the dedicated script included in the submodule:
```
cd data/UAVVaste
python main.py
```


## Start

1. Downloaded the UAVVaste dataset.

2. Generate the training and test set by running :
```
python uavvaste_save.py
```
Be careful that the #Part 1 code of the main function in decommented

3. If you want to generate waste at random places in the images run the 	#Part 2 of the main.

4. Run the training :
```
python3 GAN_mask_4/train.py --dataroot /home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_mask --name biowaste_new --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode new --use_spp --no_lsgan --norm batch --gpu_id 0 --display_id 1 --niter 100 --niter_decay 100 --input_nc 4 --output_nc 4 --n_layers_D 4 --mask True --step_opti_G 3
```

5. Test the model by using the dataset create in 2. or 3. :
```
python GAN_mask_4/test.py  -dataroot /home/ccamille/biowaste_GAN/LSP5_compost/UAVVaste_data_mask --name biowaste_mn_one_TG3 --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode new --norm batch --gpu_ids 0 --display_id 1 --input_nc 4 --output_nc 4 --n_layers_D 4 --mask True
```

4. Run 	# Part 3 of uavvaste_save.py main function to paste the generated waste on the original uavvaste images.


## Repository structure

| Path | Description
| :--- | :----------
| LSP5_compost | Repository root folder of the uavvaste GAN project.
| &boxvr;&nbsp; checkpoints | Folder with saved training experiment.
| &boxvr;&nbsp; coco_api | contains dataclasses and utilities for loading, structuring and accessing annotations in COCO format.
| &boxvr;&nbsp; GAN | Folder with the GAN algorithm.
| &boxvr;&nbsp; GAN_mask | Folder with the GAN algorithm adapted for 4 channels images.
| &boxvr;&nbsp; results | Saved trained models
| &boxvr;&nbsp; analysis.ipynb | Observe similarities between generated images and the training set.
| &boxvr;&nbsp; pre_processing.ipynb | Draft. 
| &boxvr;&nbsp; Generating_data_with_GAN.pdf | Presentation of the project.
| &boxvr;&nbsp; README.py | Description and requirement of the project.
| &boxur;&nbsp; uavvaste_save.py | Do all the pre-processing steps.
