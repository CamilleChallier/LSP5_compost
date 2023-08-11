# PS-GAN
This is a Pytorch implementation of [Pedestrian-Synthesis-GAN](https://github.com/yifanjiang19/Pedestrian-Synthesis-GAN) on UAVVaste dataset.

## Introduction

We propose an augmentation method to generate RGB + mask pairs with a GAN.

## Usage
This project uses dominate; start by install dominate and then dependencies.

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
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; base_data_loader.py | Base class for defining different dataloaders.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; base_dataset.py | Define a class that integrates the preprocessing steps for the images of the dataset.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; custom_dataset_data_loader.py | Define a base class that Create the Dataset.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; data_loader.py | Only call the initilizion of CustomDatasetDataLoader.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; image_folder.py | Define the loading of images, a class ImageFolder that integrates preprocessing steps for the images of a dataset, enabling customization of loading behavior and transformations.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; test_dataset.py | Define SingleDataset class that manages the loading, processing, and augmentation of simgle image datasets for generated waste instances. Not used in this project.
| &boxv;&nbsp; &boxvr;&nbsp; &boxur;&nbsp; train_dataset.py | Define NewDataset class that manages the loading, processing, and augmentation of image datasets for generated waste instances.
| &boxv;&nbsp; &boxvr;&nbsp; models | Folder with files that defined pix2pix model and network architecture.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; base_model.py | Base class for defining different models.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; models.py | Only call the initilizator of the Model. 
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; networks.ps | This file contains functions and classes related to the definition and initialization of generators and discriminators for a Generative Adversarial Network (GAN), including weight initialization, generator and discriminator architectures, loss functions, and spatial pyramid pooling for person detection.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; pix2pix_model.py | Defines a Pix2Pix model class that implements U-Net-based image-to-image translation with discriminators, including specialized components for tasks like person segmentation, and it also includes methods for initialization, training, and saving the model.
| &boxv;&nbsp; &boxvr;&nbsp; &boxur;&nbsp; test_model.py | Define a TestModel. Not used here.
| &boxv;&nbsp; &boxvr;&nbsp; util | Folder with useful functions.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; get_data.py | Defines a class called "GetData" that allows the downloading of datasets for CycleGAN or Pix2Pix image translation techniques, offering options for different datasets and saving the downloaded data to a specified directory.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; html.py | Defines a class named "HTML" used to generate an HTML webpage containing images, headers, and tables, and allows the user to add content like images, headers, and tables with corresponding links and text, saving the resulting webpage to a specified directory.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; image_pool.py | Defines a class called "ImagePool" that implements an image buffer for storing previously generated images, enabling updates to discriminators using a history of images rather than just the latest ones.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; png.py | Define some functions useful for images manipulation.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; util.py | This file provides utility functions for various tasks such as converting tensors to images, diagnosing networks, saving images, printing information about objects, and creating directories.
| &boxv;&nbsp; &boxvr;&nbsp; &boxur;&nbsp; visualizer.py | This file defines a Visualizer class that provides methods for displaying and saving images, plotting and printing error information, and saving images to disk, mainly used for visualization during the training process.
| &boxv;&nbsp; &boxvr;&nbsp; test.py | Define the augmentation on the test set.
| &boxv;&nbsp; &boxur;&nbsp; train.py | Define the training of the network.
| &boxvr;&nbsp; init.py | Expose the module containing the class at the level of the containing directory by importing it.
| &boxvr;&nbsp; preprocessing.py | Contains the preprocessing class useful for dataset creation.
| &boxvr;&nbsp; PSGANAugmentation.ps | Contains the interface that the augmentation method implement.
| &boxur;&nbsp; README.py | Description and requirement of the project.
