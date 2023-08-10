# PS-GAN
This is a Pytorch implementation of PS-GAN network on UAVVaste data ( credit : 
[Pedestrian-Synthesis-GAN](https://github.com/yifanjiang19/Pedestrian-Synthesis-GAN) <br /> )

## Introduction

We propose a Poisson blending loss that achieves the same purpose of Poisson Image Editing. We jointly optimize the proposed Poisson blending loss with style and content loss computed from a deep network, and reconstruct the blending region by iteratively updating the pixels using the L-BFGS solver. In the blending image, we not only smooth out gradient domain of the blending boundary but also add consistent texture into the blending region.

<img src='demo_imgs/first_demo.png' align="middle" width=540>

## Usage
This project uses [poetry](https://python-poetry.org/) to manage dependencies; start by install poetry and then dependencies

```bash
pip install poetry
poetry install
```
