"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
script aligns the VO trajectory to GT trajectory by least squares,
and saves that information
"""
import set_paths
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import os
from common.pose_utils import align_camera_poses, process_poses
import pickle
from dataset_loaders.robotcar_sdk.interpolate_poses import interpolate_ins_poses,\
  interpolate_vo_poses

# config
parser = argparse.ArgumentParser(description='Align VO poses to ground truth')
parser.add_argument('--dataset', type=str, choices=('7Scenes',
                                                    'RobotCar'), help='Dataset')
parser.add_argument('--vo_lib', type=str, choices=('dso', 'stereo', 'gps'),
                    required=True)
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--output', type=str, default=None,
  help='Output image filename')
parser.add_argument('--subsample', type=int, default=10,
  help='subsample factor for visualization')
parser.add_argument('--seq', type=str,
  help='sequence identifier e.g. 1, 2, etc for 7Scenes or 2014-06-26-08-53-56 '
       'for Robotcar')
args = parser.parse_args()
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
aux_data_dir = osp.join('..', 'data', args.dataset)

# fill the real_poses, gt_poses and frame_idx variables
if args.dataset == '7Scenes':
  assert args.vo_lib == 'dso'
  real_pose_filename = osp.join(aux_data_dir, args.scene,
                                '{:s}_poses'.format(args.vo_lib),
                                'seq-{:02d}.txt'.format(int(args.seq)))
  real_poses = np.loadtxt(real_pose_filename)
  frame_idx, real_poses = real_poses[:, 0].astype(int), real_poses[:, 1:13]
  seq_dir = osp.join(data_dir, args.scene, 'seq-{:02d}'.format(int(args.seq)))
  p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                 n.find('pose') >= 0]
  gt_poses = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
    format(i))).flatten()[:12] for i in xrange(len(p_filenames))]
  gt_poses = np.asarray(gt_poses)
elif args.dataset == 'RobotCar':
  seq_dir = osp.join(data_dir, args.scene, args.seq)
  ts_filename = osp.join(seq_dir, 'stereo.timestamps')
  with open(ts_filename, 'r') as f:
    ts = [int(l.rstrip().split(' ')[0]) for l in f]
  if args.vo_lib == 'stereo':
    vo_filename = osp.join(seq_dir, 'vo', 'vo.csv')
    real_poses = np.asarray(interpolate_vo_poses(vo_filename, ts, ts[0]))
  elif args.vo_lib == 'gps':
    vo_filename = osp.join(seq_dir, 'gps', 'gps_ins.csv')
    real_poses = np.asarray(interpolate_ins_poses(vo_filename, ts, ts[0]))
  else:
    raise NotImplementedError
  real_poses = np.asarray(real_poses)
  real_poses = np.reshape(real_poses[:, :3, :], (len(real_poses), -1))
  pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')
  gt_poses = interpolate_ins_poses(pose_filename, ts, ts[0])
  gt_poses = np.asarray(gt_poses)
  gt_poses = np.reshape(gt_poses[:, :3, :], (len(gt_poses), -1))
  assert len(real_poses) == len(gt_poses) - 1
  frame_idx = range(1, len(gt_poses))
else:
  raise NotImplementedError

gt_poses = gt_poses[frame_idx]

# calculate alignment
o1 = real_poses[:, [3, 7, 11]].copy()
o2 = gt_poses[:, [3, 7, 11]].copy()
R1 = real_poses[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]].copy().reshape((-1, 3, 3))
R2 = gt_poses[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]].copy().reshape((-1, 3, 3))
align_R, align_t, align_s = align_camera_poses(o1.T, o2.T, R1, R2)
align_t = align_t.squeeze()

# save alignment
data_dir = osp.join('..', 'data', args.dataset)
if args.dataset == '7Scenes':
  seq_dir = osp.join(data_dir, args.scene, 'seq-{:02d}'.format(int(args.seq)))
  vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(args.vo_lib))
elif args.dataset == 'RobotCar':
  seq_dir = osp.join(data_dir, args.scene, args.seq)
  vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(args.vo_lib))
else:
  raise NotImplementedError
vo_stats = {'R': align_R, 't': align_t, 's': align_s}
with open(vo_stats_filename, 'wb') as f:
  pickle.dump(vo_stats, f)
print '{:s} saved.'.format(vo_stats_filename)

# apply alignment
pose_stats_filename = osp.join(data_dir, args.scene, 'pose_stats.txt')
mean_t, std_t = np.loadtxt(pose_stats_filename)

real_poses = process_poses(real_poses, mean_t=mean_t, std_t=std_t,
  align_R=align_R, align_t=align_t, align_s=align_s)
gt_poses = process_poses(gt_poses, mean_t=mean_t, std_t=std_t, align_R=np.eye(3),
  align_t=np.zeros(3), align_s=1)

# loop
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for r,g in zip(real_poses[::args.subsample], gt_poses[::args.subsample]):
  ax.scatter(r[0], r[1], zs=r[2], c='r')
  ax.scatter(g[0], g[1], zs=g[2], c='g')

  pp = np.vstack((r, g))
  ax.plot(pp[:, 0], pp[:, 1], zs=pp[:, 2], c='b')

if args.output is None:
  plt.show(block=True)
else:
  filename = osp.expanduser(args.output)
  fig.savefig(filename)
  print '{:s} saved'.format(filename)
