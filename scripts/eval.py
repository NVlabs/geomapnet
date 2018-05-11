"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import set_paths
from models.posenet import PoseNet, MapNet
from common.train import load_state_dict, step_feedfwd
from common.pose_utils import optimize_poses, quaternion_angular_error, qexp,\
  calc_vos_safe_fc, calc_vos_safe
from dataset_loaders.composite import MF
import argparse
import os
import os.path as osp
import sys
import numpy as np
import matplotlib
DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
  matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, models
import cPickle

# config
parser = argparse.ArgumentParser(description='Evaluation script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', 'RobotCar'),
                    help='Dataset')
parser.add_argument('--scene', type=str, help='Scene name')
parser.add_argument('--weights', type=str, help='trained weights to load')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++'),
  help='Model to use (mapnet includes both MapNet and MapNet++ since their'
       'evluation process is the same and they only differ in the input weights'
       'file')
parser.add_argument('--device', type=str, default='0', help='GPU device(s)')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--val', action='store_true', help='Plot graph for val')
parser.add_argument('--output_dir', type=str, default=None,
  help='Output image directory')
parser.add_argument('--pose_graph', action='store_true',
  help='Turn on Pose Graph Optimization')
args = parser.parse_args()
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
seed = settings.getint('training', 'seed')
section = settings['hyperparameters']
dropout = section.getfloat('dropout')
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  steps = section.getint('steps')
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  fc_vos = args.dataset == 'RobotCar'
  if args.pose_graph:
    vo_lib = section.get('vo_lib')
    sax = section.getfloat('s_abs_trans', 1)
    saq = section.getfloat('s_abs_rot', 1)
    srx = section.getfloat('s_rel_trans', 20)
    srq = section.getfloat('s_rel_rot', 20)

# model
feature_extractor = models.resnet34(pretrained=False)
posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  model = MapNet(mapnet=posenet)
else:
  model = posenet
model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

# load weights
weights_filename = osp.expanduser(args.weights)
if osp.isfile(weights_filename):
  loc_func = lambda storage, loc: storage
  checkpoint = torch.load(weights_filename, map_location=loc_func)
  load_state_dict(model, checkpoint['model_state_dict'])
  print 'Loaded weights from {:s}'.format(weights_filename)
else:
  print 'Could not load weights from {:s}'.format(weights_filename)
  sys.exit(-1)

data_dir = osp.join('..', 'data', args.dataset)
stats_filename = osp.join(data_dir, args.scene, 'stats.txt')
stats = np.loadtxt(stats_filename)
# transformer
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.ToTensor(),
  transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(data_dir, args.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

# dataset
train = not args.val
if train:
  print 'Running {:s} on TRAIN data'.format(args.model)
else:
  print 'Running {:s} on VAL data'.format(args.model)
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, train=train,
  transform=data_transform, target_transform=target_transform, seed=seed)
if (args.model.find('mapnet') >= 0) or args.pose_graph:
  if args.pose_graph:
    assert real
    kwargs = dict(kwargs, vo_lib=vo_lib)
  vo_func = calc_vos_safe_fc if fc_vos else calc_vos_safe
  data_set = MF(dataset=args.dataset, steps=steps, skip=skip, real=real,
                variable_skip=variable_skip, include_vos=args.pose_graph,
                vo_func=vo_func, no_duplicates=False, **kwargs)
  L = len(data_set.dset)
elif args.dataset == '7Scenes':
  from dataset_loaders.seven_scenes import SevenScenes
  data_set = SevenScenes(**kwargs)
  L = len(data_set)
elif args.dataset == 'RobotCar':
  from dataset_loaders.robotcar import RobotCar
  data_set = RobotCar(**kwargs)
  L = len(data_set)
else:
  raise NotImplementedError

# loader (batch_size MUST be 1)
batch_size = 1
assert batch_size == 1
loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                    num_workers=5, pin_memory=True)

# activate GPUs
CUDA = torch.cuda.is_available()
torch.manual_seed(seed)
if CUDA:
  torch.cuda.manual_seed(seed)
  model.cuda()

pred_poses = np.zeros((L, 7))  # store all predicted poses
targ_poses = np.zeros((L, 7))  # store all target poses

# inference loop
for batch_idx, (data, target) in enumerate(loader):
  if batch_idx % 200 == 0:
    print 'Image {:d} / {:d}'.format(batch_idx, len(loader))

  # indices into the global arrays storing poses
  if (args.model.find('vid') >= 0) or args.pose_graph:
    idx = data_set.get_indices(batch_idx)
  else:
    idx = [batch_idx]
  idx = idx[len(idx) / 2]

  # output : 1 x 6 or 1 x STEPS x 6
  _, output = step_feedfwd(data, model, CUDA, train=False)
  s = output.size()
  output = output.cpu().data.numpy().reshape((-1, s[-1]))
  target = target.numpy().reshape((-1, s[-1]))
  
  # normalize the predicted quaternions
  q = [qexp(p[3:]) for p in output]
  output = np.hstack((output[:, :3], np.asarray(q)))
  q = [qexp(p[3:]) for p in target]
  target = np.hstack((target[:, :3], np.asarray(q)))

  if args.pose_graph:  # do pose graph optimization
    kwargs = {'sax': sax, 'saq': saq, 'srx': srx, 'srq': srq}
    # target includes both absolute poses and vos
    vos = target[len(output):]
    target = target[:len(output)]
    output = optimize_poses(pred_poses=output, vos=vos, fc_vos=fc_vos, **kwargs)

  # un-normalize the predicted and target translations
  output[:, :3] = (output[:, :3] * pose_s) + pose_m
  target[:, :3] = (target[:, :3] * pose_s) + pose_m

  # take the middle prediction
  pred_poses[idx, :] = output[len(output)/2]
  targ_poses[idx, :] = target[len(target)/2]

# calculate losses
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3],
                                                       targ_poses[:, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:],
                                                       targ_poses[:, 3:])])
#eval_func = np.mean if args.dataset == 'RobotCar' else np.median
#eval_str  = 'Mean' if args.dataset == 'RobotCar' else 'Median'
#t_loss = eval_func(t_loss)
#q_loss = eval_func(q_loss)
#print '{:s} error in translation = {:3.2f} m\n' \
#      '{:s} error in rotation    = {:3.2f} degrees'.format(eval_str, t_loss,
print 'Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
    'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(np.median(t_loss), np.mean(t_loss),
                    np.median(q_loss), np.mean(q_loss))


# create figure object
fig = plt.figure()
if args.dataset != '7Scenes':
  ax = fig.add_subplot(111)
else:
  ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

# plot on the figure object
ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
# scatter the points and draw connecting line
x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
if args.dataset != '7Scenes':  # 2D drawing
  ax.plot(x, y, c='b')
  ax.scatter(x[0, :], y[0, :], c='r')
  ax.scatter(x[1, :], y[1, :], c='g')
else:
  z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
  for xx, yy, zz in zip(x.T, y.T, z.T):
    ax.plot(xx, yy, zs=zz, c='b')
  ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
  ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
  ax.view_init(azim=119, elev=13)

if DISPLAY:
  plt.show(block=True)

if args.output_dir is not None:
  model_name = args.model
  if args.weights.find('++') >= 0:
    model_name += '++'
  if args.pose_graph:
    model_name += '_pgo_{:s}'.format(vo_lib)
  experiment_name = '{:s}_{:s}_{:s}'.format(args.dataset, args.scene, model_name)
  image_filename = osp.join(osp.expanduser(args.output_dir),
    '{:s}.png'.format(experiment_name))
  fig.savefig(image_filename)
  print '{:s} saved'.format(image_filename)
  result_filename = osp.join(osp.expanduser(args.output_dir), '{:s}.pkl'.
    format(experiment_name))
  with open(result_filename, 'wb') as f:
    cPickle.dump({'targ_poses': targ_poses, 'pred_poses': pred_poses}, f)
  print '{:s} written'.format(result_filename)
