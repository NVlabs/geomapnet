"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
from runpy import run_module
from mock import patch
import sys
import os.path as osp

def start_server(logdir, port):
  """
  Starts the Visdom server (worker function intended for multiprocessing)
  :param logdir: logging directory to save the envs
  :param port: port to run Visdom server
  :return: 
  """
  with patch.object(sys, 'argv', ['visdom.server', '-port', '{:d}'.format(port),
                                  '-env_path', '{:s}/'.format(logdir)]):
    print 'Running the Visdom server with logdir={:s} on port {:d}'.\
      format(logdir, port)
    log_filename = osp.join(logdir, 'visdom_log.txt')
    with open(log_filename, 'w') as sys.stderr:
      run_module('visdom.server', run_name='__main__')
