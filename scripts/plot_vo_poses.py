"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
script to plot the poses from integration of VO against GT poses for debugging
"""
import set_paths
from dataset_loaders.composite import OnlyPoses
from common.pose_utils import quaternion_angular_error, qexp
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
import os
import os.path as osp
import cPickle

# config
parser = argparse.ArgumentParser(description='Plot VO poses and calculate error')
parser.add_argument('--dataset', type=str, choices=('7Scenes',
                                                    'RobotCar'), help='Dataset')
parser.add_argument('--vo_lib', type=str, choices=('orbslam', 'libviso2', 'dso',
                                                   'gps', 'stereo'),
                    help='VO library to use', required=True)
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--output_dir', type=str, default=None,
  help='Output directory')
parser.add_argument('--subsample', type=int, default=10,
  help='subsample factor')
args = parser.parse_args()
data_dir = osp.join('..', 'data', args.dataset)

# read mean and stdev for un-normalizing poses
pose_stats_file = osp.join(data_dir, args.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# error criterions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

# dataset loader
train = not args.val
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
              vo_lib=args.vo_lib)
dset = OnlyPoses(dataset=args.dataset, **kwargs)

# loader
batch_size = 25
loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4)

# collect poses and losses
real_pose = np.empty((0, 6))
gt_pose = np.empty((0, 6))
for (rp, gp) in loader:
  assert len(rp) == len(gp)
  real_pose = np.vstack((real_pose, rp.numpy()))
  gt_pose = np.vstack((gt_pose, gp.numpy()))

# un-normalize and convert to quaternion
real_pose[:, :3] = (real_pose[:, :3] * pose_s) + pose_m
gt_pose[:, :3] = (gt_pose[:, :3] * pose_s) + pose_m
q = [qexp(p[3:]) for p in real_pose]
real_pose = np.hstack((real_pose[:, :3], np.asarray(q)))
q = [qexp(p[3:]) for p in gt_pose]
gt_pose = np.hstack((gt_pose[:, :3], np.asarray(q)))

# visualization loop
T = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
for r,g in zip(real_pose[::args.subsample], gt_pose[::args.subsample]):
  ax.scatter(r[0], r[1], zs=r[2], c='r')
  ax.scatter(g[0], g[1], zs=g[2], c='g')
  pp = np.vstack((r, g))
  ax.plot(pp[:, 0], pp[:, 1], zs=pp[:, 2], c='b')
  ax.view_init(azim=-137, elev=52)

# error calculation loop
t_loss = []
q_loss = []
for (r, g) in zip(real_pose, gt_pose):
  t_loss.append(t_criterion(r[:3], g[:3]))
  q_loss.append(q_criterion(r[3:], g[3:]))
t_loss = np.mean(np.asarray(t_loss))
q_loss = np.mean(np.asarray(q_loss))
print 'Median t-Loss = {:f}, Median q-Loss = {:f}'.format(t_loss, q_loss)

if train:
  print 'Visualizing TRAIN data'
else:
  print 'Visualizing VAL data'

plt.show(block=True)
if args.output_dir is not None:
  experiment_name = '{:s}_{:s}_{:s}'.format(args.dataset, args.scene,
                                            args.vo_lib)
  image_filename = osp.join(osp.expanduser(args.output_dir),
    '{:s}.png'.format(experiment_name))
  fig.savefig(image_filename)
  print '{:s} saved'.format(image_filename)
  result_filename = osp.join(osp.expanduser(args.output_dir),
    '{:s}.pkl'.format(experiment_name))
  with open(result_filename, 'wb') as f:
    cPickle.dump({'targ_poses': gt_pose, 'pred_poses': real_pose}, f)
  print '{:s} written'.format(result_filename)
