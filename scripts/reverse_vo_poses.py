"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
script reverses the poses in a sequence which needed to be reversed to run VO
"""
import numpy as np
import os.path as osp
import argparse
import os

# config
parser = argparse.ArgumentParser(description='Reverse poses')
parser.add_argument('--dataset', type=str, choices=('KITTIOdometry', '7Scenes'),
                                                    help='Dataset')
parser.add_argument('--vo_lib', type=str, choices='dso',
                    help='VO library to use', required=True)
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--seq', type=str, help='sequence identifier e.g. 1, 2, etc'
                                            'for 7Scenes or 2014-06-26-08-53-56'
                                            'for Robotcar')
args = parser.parse_args()
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
aux_data_dir = osp.join('..', 'data', args.dataset)

if args.dataset == 'KITTIOdometry':
  # find the number of frames in GT poses file
  gt_pose_filename = osp.join(data_dir, 'poses', '{:s}.txt'.format(args.scene))
  gt_poses = np.loadtxt(gt_pose_filename)
  N = len(gt_poses)

  # reverse poses
  real_pose_filename = osp.join(aux_data_dir, '{:s}_poses'.format(args.vo_lib),
    '{:s}.txt'.format(args.scene))
  real_poses = np.loadtxt(real_pose_filename)
  frame_idx, real_poses = real_poses[:, 0].astype(int), real_poses[:, 1:]
elif args.dataset == '7Scenes':
  seq_dir = osp.join(data_dir, args.scene, 'seq-{:02d}'.format(int(args.seq)))
  # find the number of frames in GT poses file
  p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                 n.find('pose') >= 0]
  gt_poses = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
                                  format(i))).flatten()[:12] for i in
              xrange(len(p_filenames))]
  N = len(gt_poses)

  # reverse poses
  real_pose_filename = osp.join(aux_data_dir, args.scene,
                                '{:s}_poses'.format(args.vo_lib),
                                'seq-{:02d}.txt'.format(int(args.seq)))
  real_poses = np.loadtxt(real_pose_filename)
  frame_idx, real_poses = real_poses[:, 0].astype(int), real_poses[:, 1:]
else:
  raise NotImplementedError

assert max(frame_idx) < N

fmt = '%d ' + '%8.7f '*real_poses.shape[1]
frame_idx = N - 1 - frame_idx
order = np.argsort(frame_idx)
out_data = np.hstack((frame_idx[order, np.newaxis], real_poses[order]))
# out_filename = real_pose_filename[:-4] + '_reversed' + real_pose_filename[-4:]
out_filename = real_pose_filename
np.savetxt(out_filename, out_data, fmt=fmt)
print '{:s} saved'.format(out_filename)
