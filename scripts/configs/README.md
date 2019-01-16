[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
# MapNet: Geometry-Aware Learning of Maps for Camera Localization 

## License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 


## Hyperparameters
Most parameters in the config files are self-explanatory. Here are some notes:

- `beta` is the initial value of Beta in Eq. 3 of the paper

- `gamma` is the initial value of Gamma in Eq. 3 of the paper

- `steps` is the size of tuples of images to use for training (parameter `s`
described in Section 3.5 of the paper)

- `skip` is the spacing between these images (parameter `k` described in
Section 3.5 of the paper)

- `real` is a flag indicating whether the poses should be GPS/SLAM/integration
of visual odometry (true) or from ground truth (false)

- `color_jitter` is the intensity of color jittering (brightness, hue, contrast and saturation) data augmentation.
NOTE: Set `color_jitter = 0` in `mapnet.ini` while training it on the 7 Scenes dataset. 

- `s_abs_trans`, `s_abs_rot`, `s_rel_trans`, `s_rel_rot` are the covariance
values for absolute and relative translations and rotations passed to the PGO
algorithm (see Appendix in our
[arXiv paper](https://arxiv.org/pdf/1712.03342.pdf)). To reproduce results from
our paper, use the following values:

7 Scenes:

Scene | `s_abs_trans`| `s_abs_rot`| `s_rel_trans`| `s_rel_rot`
---|:---:|:---:|:---:|:---:
chess | 1 | 1 | 35 | 35
fire | 1 | 1 | 10 | 10
heads | 1 | 1 | 1 | 1
office | 0.1 | 0.1 | 20 | 10
pumpkin | 1 | 1 | 500 | 500
redkitchen | 1 | 1 | 35 | 35
stairs | 1 | 1 | 2 | 4

RobotCar

Scene | `s_abs_trans`| `s_abs_rot`| `s_rel_trans`| `s_rel_rot`
---|:---:|:---:|:---:|:---:
loop | 1 | 1 | 20 | 20
full | 1 | 1 | 1 | 10

## Which files to use?
### Training
Use the files according to their name e.g. use `mapnet++_7Scenes.ini` if you
want to train a MapNet++ model on the 7 Scenes dataset.
### Inference
If you want to perform pose-graph optimization (PGO) during inference on any 
of the trained models, use the `pgo_inference_*.ini` files. The inference of a
MapNet++ model (without PGO) should be done with `mapnet.ini`.
