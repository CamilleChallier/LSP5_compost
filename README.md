# PS-GAN
This is a Pytorch implementation of [Pedestrian-Synthesis-GAN](https://github.com/yifanjiang19/Pedestrian-Synthesis-GAN) on UAVVaste data

## Introduction

We propose a Poisson blending loss that achieves the same purpose of Poisson Image Editing. We jointly optimize the proposed Poisson blending loss with style and content loss computed from a deep network, and reconstruct the blending region by iteratively updating the pixels using the L-BFGS solver. In the blending image, we not only smooth out gradient domain of the blending boundary but also add consistent texture into the blending region.

## Usage
This project uses dominate; start by install dominate and then dependencies

```bash
pip install dominate
```

##
