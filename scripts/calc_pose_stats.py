"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
script to calculate pose translation stats (run first for every dataset)
"""
import set_paths
from dataset_loaders.seven_scenes import SevenScenes
from dataset_loaders.robotcar import RobotCar
import argparse
import os.path as osp

# config
parser = argparse.ArgumentParser(description='Calculate pose translation stats')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
args = parser.parse_args()
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)

# dataset loader
# creating the dataset with train=True and real=False saves the stats from the
# training split
kwargs = dict(sequence=args.scene, data_path=data_dir, train=True, real=False,
  skip_images=True, seed=7)
if args.dataset == '7Scenes':
  dset = SevenScenes(**kwargs)
elif args.dataset == 'RobotCar':
  dset = RobotCar(**kwargs)
else:
  raise NotImplementedError

print 'Done'
