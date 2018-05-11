"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
""""
Implementation of VidLoc: 6-DoF Video-Clip Relocalization - CVPR 2017 - Clark et al
"""
import torch
from torch import nn
from torch.nn import init, functional as F
from torch.autograd import Variable
from collections import OrderedDict
from common.pose_utils import normalize

class VidLoc(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=1024):
    super(VidLoc, self).__init__()
    n_layers = 1
    self.droprate = droprate

    # remove the last FC layer in feature extractor
    d = OrderedDict(feature_extractor.named_children())
    _, fc = d.popitem(last=True)  # remove last layer, which is fc
    fe_out_planes = fc.in_features
    self.feature_extractor = nn.Sequential(d)
    self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)

    # LSTM (dropout is useless because n_layers = 1
    self.lstm_fc = nn.LSTM(input_size=fe_out_planes, hidden_size=feat_dim,
      num_layers=n_layers, bidirectional=True, batch_first=True)
    self.lstm_xyz = nn.LSTM(input_size=2*feat_dim, hidden_size=3,
      num_layers=n_layers, bidirectional=False, batch_first=True)
    self.lstm_wpqr = nn.LSTM(input_size=2*feat_dim, hidden_size=4,
      num_layers=n_layers, bidirectional=False, batch_first=True)

    # initialize the feature extractor
    if not pretrained:
      for m in self.feature_extractor.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
          init.kaiming_normal(m.weight.data)
          if m.bias is not None:
            init.constant(m.bias.data, 0)

    # initialize the LSTM
    # "An Empirical Exploration of Recurrent Network Architectures" -
    # JMLR, Jozefowicz et al
    for layer in xrange(n_layers):
      for connection in ['ih', 'hh']:
        for suffix in ['', '_reverse']:
          bias_name = 'bias_{:s}_l{:d}{:s}'.format(connection, layer, suffix)
          b = getattr(self.lstm_fc, bias_name)
          N = b.size(0)
          init.constant(b[N/4 : (N/2)+1], 1)

    for layer in xrange(n_layers):
      for connection in ['ih', 'hh']:
        for lstm in [self.lstm_xyz, self.lstm_wpqr]:
          bias_name = 'bias_{:s}_l{:d}'.format(connection, layer)
          b = getattr(lstm, bias_name)
          N = b.size(0)
          init.constant(b[N/4 : (N/2)+1], 1)

    # hidden states
    self.fc_h = Variable(None)
    self.fc_c = Variable(None)
    self.xyz_h = Variable(None)
    self.xyz_c = Variable(None)
    self.wpqr_h = Variable(None)
    self.wpqr_c = Variable(None)

  def forward(self, x, cuda=False, async=True):
    """
    :param x: B x G x C x H x W;
    :param cuda
    :param async
    :return: poses B x G x 7
    """
    if cuda:
      x = x.cuda(async=async)
    s = x.size()
    x = x.view(-1, *s[2:])
    x = self.feature_extractor(x).squeeze()  # BG x d
    x = x.view(s[0], s[1], -1)  # B x G x d
    x, (self.fc_h, self.fc_c) = self.lstm_fc(x, (self.fc_h, self.fc_c))  # B x G x d'
    x = F.relu(x)  # B x G x d'
    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate)  # B x G x d'
    xyz, (self.xyz_h, self.xyz_c) = self.lstm_xyz(x,
      (self.xyz_h, self.xyz_c))  # B x G x 3
    wpqr, (self.wpqr_h, self.wpqr_c) = self.lstm_wpqr(x,
      (self.wpqr_h, self.wpqr_c))  # B x G x 4
    wpqr = wpqr.contiguous().view(-1, 4)
    wpqr = normalize(wpqr, dim=1, p=2)  # normalize the quaternion
    wpqr = wpqr.view(s[0], s[1], -1)

    return torch.cat((xyz, wpqr), dim=2)

  def reset_hidden_states(self, batch_size):
    self.fc_h.data.resize_(2, batch_size, self.lstm_fc.hidden_size).normal_()
    self.fc_c.data.resize_(2, batch_size, self.lstm_fc.hidden_size).normal_()
    self.xyz_h.data.resize_(1, batch_size, self.lstm_xyz.hidden_size).normal_()
    self.xyz_c.data.resize_(1, batch_size, self.lstm_xyz.hidden_size).normal_()
    self.wpqr_h.data.resize_(1, batch_size, self.lstm_wpqr.hidden_size).normal_()
    self.wpqr_c.data.resize_(1, batch_size, self.lstm_wpqr.hidden_size).normal_()

  def detach_hidden_states(self):
    self.fc_h   = Variable(self.fc_h.data)
    self.fc_c   = Variable(self.fc_c.data)
    self.xyz_h  = Variable(self.xyz_h.data)
    self.xyz_c  = Variable(self.xyz_c.data)
    self.wpqr_h = Variable(self.wpqr_h.data)
    self.wpqr_c = Variable(self.wpqr_c.data)

  def cuda(self, device_id=None):
    super(VidLoc, self).cuda(device_id=device_id)

    self.fc_c = self.fc_c.cuda(device_id=device_id)
    self.fc_h = self.fc_h.cuda(device_id=device_id)
    self.xyz_c = self.xyz_c.cuda(device_id=device_id)
    self.xyz_h = self.xyz_h.cuda(device_id=device_id)
    self.wpqr_c = self.wpqr_c.cuda(device_id=device_id)
    self.wpqr_h = self.wpqr_h.cuda(device_id=device_id)

if __name__ == '__main__':
  from torchvision import models
  from torch.autograd import Variable
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  feature_extractor = models.resnet18(pretrained=False)
  v = VidLoc(feature_extractor).cuda()
  x = Variable(torch.rand(10, 10, 3, 240, 320)).cuda()

  y = v(x)
  pass
