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
