"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Script to demosaic and undistort images and save them in a 'centre_processed'
folder.
The RobotCar dataset must use the SDK image loader function for this script to
work!
"""
import set_paths
from dataset_loaders.robotcar import RobotCar
import argparse
import os.path as osp
from torch.utils.data import DataLoader
from common.train import safe_collate
from torchvision import transforms
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Demosaic and undistort RobotCar '
                                             'images')
parser.add_argument('--scene', type=str, required=True)
parser.add_argument('--n_cores', type=int, default=4)
parser.add_argument('--val', action='store_true')
args = parser.parse_args()
if args.val:
  print 'processing VAL data using {:d} cores'.format(args.n_cores)
else:
  print 'processing TRAIN data using {:d} cores'.format(args.n_cores)

# create data loader
batch_size = args.n_cores
data_dir = osp.join('..', 'data', 'deepslam_data', 'RobotCar')
transform = transforms.Compose([transforms.Scale(256),
                                transforms.Lambda(lambda x: np.asarray(x))])
dset = RobotCar(sequence=args.scene, data_path=data_dir, train=(not args.val),
                transform=transform, undistort=True)
loader = DataLoader(dset, batch_size=batch_size, num_workers=args.n_cores,
                    collate_fn=safe_collate)

# gather information about output filenames
base_dir = osp.join(data_dir, args.scene)
if args.val:
  split_filename = osp.join(base_dir, 'test_split.txt')
else:
  split_filename = osp.join(base_dir, 'train_split.txt')
with open(split_filename, 'r') as f:
  seqs = [l.rstrip() for l in f if not l.startswith('#')]

im_filenames = []
for seq in seqs:
  seq_dir = osp.join(base_dir, seq)
  ts_filename = osp.join(seq_dir, 'stereo.timestamps')
  with open(ts_filename, 'r') as f:
    ts = [l.rstrip().split(' ')[0] for l in f]
  im_filenames.extend([osp.join(seq_dir, 'stereo', 'centre_processed', '{:s}.png'.
                                format(t)) for t in ts])
assert len(dset) == len(im_filenames)

# loop
for batch_idx, (imgs, _) in enumerate(loader):
  for idx, im in enumerate(imgs):
    im_filename = im_filenames[batch_idx*batch_size + idx]
    im = Image.fromarray(im.numpy())
    try:
      im.save(im_filename)
    except IOError:
      print 'IOError while saving {:s}'.format(im_filename)

  if batch_idx % 50 == 0:
    print 'Processed {:d} / {:d}'.format(batch_idx*batch_size, len(dset))
