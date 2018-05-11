"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
pytorch data loader for the 7-scenes dataset
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data
from utils import load_image
import sys
import pickle

sys.path.insert(0, '../')
from common.pose_utils import process_poses

class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7, real=False,
                 skip_images=False, vo_lib='orbslam'):
      """
      :param scene: scene name ['chess', 'pumpkin', ...]
      :param data_path: root 7scenes data directory.
      Usually '../data/deepslam_data/7Scenes'
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the images
      :param target_transform: transform to apply to the poses
      :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
      :param real: If True, load poses from SLAM/integration of VO
      :param skip_images: If True, skip loading images and return None instead
      :param vo_lib: Library to use for VO (currently only 'dso')
      """
      self.mode = mode
      self.transform = transform
      self.target_transform = target_transform
      self.skip_images = skip_images
      np.random.seed(seed)

      # directories
      base_dir = osp.join(osp.expanduser(data_path), scene)
      data_dir = osp.join('..', 'data', '7Scenes', scene)

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'TrainSplit.txt')
      else:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}
      vo_stats = {}
      gt_offset = int(0)
      for seq in seqs:
        seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
        seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
        if real:
          pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib),
                               'seq-{:02d}.txt'.format(seq))
          pss = np.loadtxt(pose_file)
          frame_idx = pss[:, 0].astype(np.int)
          if vo_lib == 'libviso2':
            frame_idx -= 1
          ps[seq] = pss[:, 1:13]
          vo_stats_filename = osp.join(seq_data_dir,
                                       '{:s}_vo_stats.pkl'.format(vo_lib))
          with open(vo_stats_filename, 'rb') as f:
            vo_stats[seq] = pickle.load(f)
          # # uncomment to check that PGO does not need aligned VO!
          # vo_stats[seq]['R'] = np.eye(3)
          # vo_stats[seq]['t'] = np.zeros(3)
        else:
          frame_idx = np.array(xrange(len(p_filenames)), dtype=np.int)
          pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
            format(i))).flatten()[:12] for i in frame_idx]
          ps[seq] = np.asarray(pss)
          vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
        gt_offset += len(p_filenames)
        c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                  for i in frame_idx]
        d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                  for i in frame_idx]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)

      pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
      if train and not real:
        mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
        std_t = np.ones(3)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
      else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

      # convert pose to translation + log quaternion
      self.poses = np.empty((0, 6))
      for seq in seqs:
        pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
          align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
          align_s=vo_stats[seq]['s'])
        self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
      if self.skip_images:
        img = None
        pose = self.poses[index]
      else:
        if self.mode == 0:
          img = None
          while img is None:
            img = load_image(self.c_imgs[index])
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 1:
          img = None
          while img is None:
            img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 2:
          c_img = None
          d_img = None
          while (c_img is None) or (d_img is None):
            c_img = load_image(self.c_imgs[index])
            d_img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
          img = [c_img, d_img]
          index -= 1
        else:
          raise Exception('Wrong mode {:d}'.format(self.mode))

      if self.target_transform is not None:
        pose = self.target_transform(pose)

      if self.skip_images:
        return img, pose

      if self.transform is not None:
        if self.mode == 2:
          img = [self.transform(i) for i in img]
        else:
          img = self.transform(img)

      return img, pose

    def __len__(self):
      return self.poses.shape[0]

def main():
  """
  visualizes the dataset
  """
  from common.vis_utils import show_batch, show_stereo_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  seq = 'chess'
  mode = 2
  num_workers = 6
  transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225])
  ])
  dset = SevenScenes(seq, '../data/deepslam_data/7Scenes', True, transform,
    mode=mode)
  print 'Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq,
    len(dset))

  data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
    num_workers=num_workers)

  batch_count = 0
  N = 2
  for batch in data_loader:
    print 'Minibatch {:d}'.format(batch_count)
    if mode < 2:
      show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
    elif mode == 2:
      lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
      rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
      show_stereo_batch(lb, rb)

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
