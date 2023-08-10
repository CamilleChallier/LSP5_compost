# PS-GAN
This is a Pytorch implementation of [Pedestrian-Synthesis-GAN](https://github.com/yifanjiang19/Pedestrian-Synthesis-GAN) on UAVVaste dataset.

## Introduction

We propose an augmentation method to generate RGB + mask pairs with a GAN.

## Usage
This project uses dominate; start by install dominate and then dependencies

```bash
pip install dominate
```

(opt) This project use visdom for visualization. Run it to see the training process.

```bash
pip install visdom
python -m visdom.server
```

## Repository structure


| Path | Description
| :--- | :----------
| PS_GAN | Repository root folder of the uavvaste GAN project.
| &boxvr;&nbsp; checkpoints | Folder with saved training experiment.
| &boxvr;&nbsp; GAN_mask | Folder with the GAN algorithm.
| &boxv;&nbsp; &boxvr;&nbsp; data | Folder with dataset loader and data pre-processing.
| &boxv;&nbsp; &boxvr;&nbsp; models | Folder with files that defined pix2pix model and network architecture.
| &boxv;&nbsp; &boxvr;&nbsp; util | Folder with useful functions.
| &boxv;&nbsp; &boxur;&nbsp; test.py | Define the augmentation on the test set.
| &boxv;&nbsp; &boxur;&nbsp; train.py | Define the training of the network.
| &boxvr;&nbsp; init.py | Expose the module containing the class at the level of the containing directory by importing it.
| &boxvr;&nbsp; preprocessing.py | Contains the preprocessing class useful for dataset creation.
| &boxvr;&nbsp; PSGANAugmentation.ps | Contains the interface that the augmentation method implement.
| &boxvr;&nbsp; README.py | Description and requirement of the project.
