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
| ALAE | Repository root folder
| &boxvr;&nbsp; configs | Folder with yaml config files.
| &boxv;&nbsp; &boxvr;&nbsp; bedroom.yaml | Config file for LSUN bedroom dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba.yaml | Config file for CelebA dataset at 128x128 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba-hq256.yaml | Config file for CelebA-HQ dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_nostyle.yaml | Config file for CelebA 128x128 dataset for ablation study (no styles).
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_separate.yaml | Config file for CelebA 128x128 dataset for ablation study (separate encoder and discriminator).
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_z_reg.yaml | Config file for CelebA 128x128 dataset for ablation study (regress in Z space, not W).
| &boxv;&nbsp; &boxvr;&nbsp; ffhq.yaml | Config file for FFHQ dataset at 1024x1024 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; mnist.yaml | Config file for MNIST dataset using Style architecture.
| &boxv;&nbsp; &boxur;&nbsp; mnist_fc.yaml | Config file for MNIST dataset using only fully connected layers (Permutation Invariant MNIST).
| &boxvr;&nbsp; dataset_preparation | Folder with scripts for dataset preparation.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_celeba_hq_tfrec.py | To prepare TFRecords for CelebA-HQ dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_celeba_tfrec.py | To prepare TFRecords for CelebA dataset at 128x128 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_mnist_tfrec.py | To prepare TFRecords for MNIST dataset.
| &boxv;&nbsp; &boxvr;&nbsp; split_tfrecords_bedroom.py | To split official TFRecords from StyleGAN paper for LSUN bedroom dataset.
| &boxv;&nbsp; &boxur;&nbsp; split_tfrecords_ffhq.py | To split official TFRecords from StyleGAN paper for FFHQ dataset.
| &boxvr;&nbsp; dataset_samples | Folder with sample inputs for different datasets. Used for figures and for test inputs during training.
| &boxvr;&nbsp; make_figures | Scripts for making various figures.
| &boxvr;&nbsp; metrics | Scripts for computing metrics.
| &boxvr;&nbsp; principal_directions | Scripts for computing principal direction vectors for various attributes. **For interactive demo**.
| &boxvr;&nbsp; style_mixing | Sample inputs and script for producing style-mixing figures.
| &boxvr;&nbsp; training_artifacts | Default place for saving checkpoints/sample outputs/plots.
| &boxv;&nbsp; &boxur;&nbsp; download_all.py | Script for downloading all pretrained models.
| &boxvr;&nbsp; interactive_demo.py | Runnable script for interactive demo.
| &boxvr;&nbsp; train_alae.py | Runnable script for training.
| &boxvr;&nbsp; train_alae_separate.py | Runnable script for training for ablation study (separate encoder and discriminator).
| &boxvr;&nbsp; checkpointer.py | Module for saving/restoring model weights, optimizer state and loss history.
| &boxvr;&nbsp; custom_adam.py | Customized adam optimizer for learning rate equalization and zero second beta.
| &boxvr;&nbsp; dataloader.py | Module with dataset classes, loaders, iterators, etc.
| &boxvr;&nbsp; defaults.py | Definition for config variables with default values.
| &boxvr;&nbsp; launcher.py | Helper for running multi-GPU, multiprocess training. Sets up config and logging.
| &boxvr;&nbsp; lod_driver.py | Helper class for managing growing/stabilizing network.
| &boxvr;&nbsp; lreq.py | Custom `Linear`, `Conv2d` and `ConvTranspose2d` modules for learning rate equalization.
| &boxvr;&nbsp; model.py | Module with high-level model definition.
| &boxvr;&nbsp; model_separate.py | Same as above, but for ablation study.
| &boxvr;&nbsp; net.py | Definition of all network blocks for multiple architectures.
| &boxvr;&nbsp; registry.py | Registry of network blocks for selecting from config file.
| &boxvr;&nbsp; scheduler.py | Custom schedulers with warm start and aggregating several optimizers.
| &boxvr;&nbsp; tracker.py | Module for plotting losses.
| &boxur;&nbsp; utils.py | Decorator for async call, decorator for caching, registry for network blocks.
