"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
implementation of PoseNet and MapNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np

import os
os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')

import sys
sys.path.insert(0, '../')

#def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)

class PoseNet(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False):
    super(PoseNet, self).__init__()
    self.droprate = droprate

    # replace the last FC layer in feature extractor
    self.feature_extractor = feature_extractor
    self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    fe_out_planes = self.feature_extractor.fc.in_features
    self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 3)
    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, x):
    x = self.feature_extractor(x)
    x = F.relu(x)
    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate)

    xyz  = self.fc_xyz(x)
    wpqr = self.fc_wpqr(x)
    return torch.cat((xyz, wpqr), 1)

class MapNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """
  def __init__(self, mapnet):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(MapNet, self).__init__()
    self.mapnet = mapnet

  def forward(self, x):
    """
    :param x: image blob (N x T x C x H x W)
    :return: pose outputs
     (N x T x 6)
    """
    s = x.size()
    x = x.view(-1, *s[2:])
    poses = self.mapnet(x)
    poses = poses.view(s[0], s[1], -1)
    return poses
