"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import pad
import numpy as np
import scipy.linalg as slin
import math
import transforms3d.quaternions as txq
import transforms3d.euler as txe
# see for formulas:
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-801-machine-vision-fall-2004/readings/quaternions.pdf
# and "Quaternion and Rotation" - Yan-Bin Jia, September 18, 2016
#from IPython.core.debugger import set_trace

## PYTORCH
def vdot(v1, v2):
  """
  Dot product along the dim=1
  :param v1: N x d
  :param v2: N x d
  :return: N x 1
  """
  out = torch.mul(v1, v2)
  out = torch.sum(out, 1)
  return out

def normalize(x, p=2, dim=0):
  """
  Divides a tensor along a certain dim by the Lp norm
  :param x: 
  :param p: Lp norm
  :param dim: Dimension to normalize along
  :return: 
  """
  xn = x.norm(p=p, dim=dim)
  x = x / xn.unsqueeze(dim=dim)
  return x

def qmult(q1, q2):
  """
  Multiply 2 quaternions
  :param q1: Tensor N x 4
  :param q2: Tensor N x 4
  :return: quaternion product, Tensor N x 4
  """
  q1s, q1v = q1[:, :1], q1[:, 1:]
  q2s, q2v = q2[:, :1], q2[:, 1:]

  qs = q1s*q2s - vdot(q1v, q2v)
  qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) +\
       torch.cross(q1v, q2v, dim=1)
  q  = torch.cat((qs, qv), dim=1)

  # normalize
  q = normalize(q, dim=1)

  return q

def qinv(q):
  """
  Inverts quaternions
  :param q: N x 4
  :return: q*: N x 4 
  """
  q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
  return q_inv

def qexp_t(q):
  """
  Applies exponential map to log quaternion
  :param q: N x 3
  :return: N x 4
  """
  n = torch.norm(q, p=2, dim=1, keepdim=True)
  n = torch.clamp(n, min=1e-8)
  q = q * torch.sin(n)
  q = q / n
  q = torch.cat((torch.cos(n), q), dim=1)
  return q

def qlog_t(q):
  """
  Applies the log map to a quaternion
  :param q: N x 4
  :return: N x 3
  """
  n = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
  n = torch.clamp(n, min=1e-8)
  q = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))
  q = q / n
  return q

def qexp_t_safe(q):
  """
  Applies exponential map to log quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 3
  :return: N x 4
  """
  q = torch.from_numpy(np.asarray([qexp(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q

def qlog_t_safe(q):
  """
  Applies the log map to a quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 4
  :return: N x 3
  """
  q = torch.from_numpy(np.asarray([qlog(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q

def rotate_vec_by_q(t, q):
  """
  rotates vector t by quaternion q
  :param t: vector, Tensor N x 3
  :param q: quaternion, Tensor N x 4
  :return: t rotated by q: t' = t + 2*qs*(qv x t) + 2*qv x (qv x r) 
  """
  qs, qv = q[:, :1], q[:, 1:]
  b  = torch.cross(qv, t, dim=1)
  c  = 2 * torch.cross(qv, b, dim=1)
  b  = 2 * b.mul(qs.expand_as(b))
  tq = t + b + c
  return tq

def compose_pose_quaternion(p1, p2):
  """
  pyTorch implementation
  :param p1: input pose, Tensor N x 7
  :param p2: pose to apply, Tensor N x 7
  :return: output pose, Tensor N x 7
  all poses are translation + quaternion
  """
  p1t, p1q = p1[:, :3], p1[:, 3:]
  p2t, p2q = p2[:, :3], p2[:, 3:]
  q = qmult(p1q, p2q)
  t = p1t + rotate_vec_by_q(p2t, p1q)
  return torch.cat((t, q), dim=1)

def invert_pose_quaternion(p):
  """
  inverts the pose
  :param p: pose, Tensor N x 7
  :return: inverted pose
  """
  t, q = p[:, :3], p[:, 3:]
  q_inv = qinv(q)
  tinv = -rotate_vec_by_q(t, q_inv)
  return torch.cat((tinv, q_inv), dim=1)

def calc_vo(p0, p1):
  """
  calculates VO (in the p0 frame) from 2 poses
  :param p0: N x 7
  :param p1: N x 7
  """
  return compose_pose_quaternion(invert_pose_quaternion(p0), p1)

def calc_vo_logq(p0, p1):
  """
  VO (in the p0 frame) (logq)
  :param p0: N x 6
  :param p1: N x 6
  :return: N-1 x 6
  """
  q0 = qexp_t(p0[:, 3:])
  q1 = qexp_t(p1[:, 3:])
  vos = calc_vo(torch.cat((p0[:, :3], q0), dim=1), torch.cat((p1[:, :3], q1),
                                                             dim=1))
  vos_q = qlog_t(vos[:, 3:])
  return torch.cat((vos[:, :3], vos_q), dim=1)

def calc_vo_relative(p0, p1):
  """
  calculates VO (in the world frame) from 2 poses
  :param p0: N x 7
  :param p1: N x 7
  """
  vos_t = p1[:, :3] - p0[:, :3]
  vos_q = qmult(qinv(p0[:, 3:]), p1[:, 3:])
  return torch.cat((vos_t, vos_q), dim=1)

def calc_vo_relative_logq(p0, p1):
  """
  Calculates VO (in the world frame) from 2 poses (log q)
  :param p0: N x 6
  :param p1: N x 6
  :return:
  """
  q0 = qexp_t(p0[:, 3:])
  q1 = qexp_t(p1[:, 3:])
  vos = calc_vo_relative(torch.cat((p0[:, :3], q0), dim=1),
                         torch.cat((p1[:, :3], q1), dim=1))
  vos_q = qlog_t(vos[:, 3:])
  return torch.cat((vos[:, :3], vos_q), dim=1)

def calc_vo_relative_logq_safe(p0, p1):
  """
  Calculates VO (in the world frame) from 2 poses (log q) through numpy fns
  :param p0: N x 6
  :param p1: N x 6
  :return:
  """
  vos_t = p1[:, :3] - p0[:, :3]
  q0 = qexp_t_safe(p0[:, 3:])
  q1 = qexp_t_safe(p1[:, 3:])
  vos_q = qmult(qinv(q0), q1)
  vos_q = qlog_t_safe(vos_q)
  return torch.cat((vos_t, vos_q), dim=1)

def calc_vo_logq_safe(p0, p1):
  """
  VO in the p0 frame using numpy fns
  :param p0:
  :param p1:
  :return:
  """
  vos_t = p1[:, :3] - p0[:, :3]
  q0 = qexp_t_safe(p0[:, 3:])
  q1 = qexp_t_safe(p1[:, 3:])
  vos_t = rotate_vec_by_q(vos_t, qinv(q0))
  vos_q = qmult(qinv(q0), q1)
  vos_q = qlog_t_safe(vos_q)
  return torch.cat((vos_t, vos_q), dim=1)

def calc_vos_simple(poses):
  """
  calculate the VOs, from a list of consecutive poses
  :param poses: N x T x 7
  :return: N x (T-1) x 7
  """
  vos = []
  for p in poses:
    pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in xrange(len(p)-1)]
    vos.append(torch.cat(pvos, dim=0))
  vos = torch.stack(vos, dim=0)

  return vos

def calc_vos(poses):
  """
  calculate the VOs, from a list of consecutive poses (in the p0 frame)
  :param poses: N x T x 7
  :return: N x (T-1) x 7
  """
  vos = []
  for p in poses:
    pvos = [calc_vo_logq(p[i].unsqueeze(0), p[i+1].unsqueeze(0))
            for i in xrange(len(p)-1)]
    vos.append(torch.cat(pvos, dim=0))
  vos = torch.stack(vos, dim=0)
  return vos

def calc_vos_relative(poses):
  """
  calculate the VOs, from a list of consecutive poses (in the world frame)
  :param poses: N x T x 7
  :return: N x (T-1) x 7
  """
  vos = []
  for p in poses:
    pvos = [calc_vo_relative_logq(p[i].unsqueeze(0), p[i+1].unsqueeze(0))
            for i in xrange(len(p)-1)]
    vos.append(torch.cat(pvos, dim=0))
  vos = torch.stack(vos, dim=0)
  return vos

def calc_vos_safe(poses):
  """
  calculate the VOs, from a list of consecutive poses
  :param poses: N x T x 7
  :return: N x (T-1) x 7
  """
  vos = []
  for p in poses:
    pvos = [calc_vo_logq_safe(p[i].unsqueeze(0), p[i+1].unsqueeze(0))
            for i in xrange(len(p)-1)]
    vos.append(torch.cat(pvos, dim=0))
  vos = torch.stack(vos, dim=0)
  return vos

def calc_vos_safe_fc(poses):
  """
  calculate the VOs, from a list of consecutive poses (fully connected)
  :param poses: N x T x 7
  :return: N x TC2 x 7
  """
  vos = []
  for p in poses:
    pvos = []
    for i in xrange(p.size(0)):
      for j in xrange(i+1, p.size(0)):
        pvos.append(calc_vo_logq_safe(p[i].unsqueeze(0), p[j].unsqueeze(0)))
    vos.append(torch.cat(pvos, dim=0))
  vos = torch.stack(vos, dim=0)
  return vos

## NUMPY
def qlog(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if all(q[1:] == 0):
    q = np.zeros(3)
  else:
    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
  return q

def qexp(q):
  """
  Applies the exponential map to q
  :param q: (3,)
  :return: (4,)
  """
  n = np.linalg.norm(q)
  q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
  return q

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :param align_R: 3 x 3
  :param align_t: 3
  :param align_s: 1
  :return: processed poses (translation + quaternion) N x 7
  """
  poses_out = np.zeros((len(poses_in), 6))
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
  for i in xrange(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3]
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere
    q = qlog(q)
    poses_out[i, 3:] = q
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t
  poses_out[:, :3] /= std_t
  return poses_out

def log_quaternion_angular_error(q1, q2):
  return quaternion_angular_error(qexp(q1), qexp(q2))

def quaternion_angular_error(q1, q2):
  """
  angular error between two quaternions
  :param q1: (4, )
  :param q2: (4, )
  :return:
  """
  d = abs(np.dot(q1, q2))
  d = min(1.0, max(-1.0, d))
  theta = 2 * np.arccos(d) * 180 / np.pi
  return theta

def skew(x):
  """
  returns skew symmetric matrix from vector
  :param x: 3 x 1
  :return:
  """
  s = np.asarray([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
  return s

def dpq_q(p):
  """
  returns the jacobian of quaternion product pq w.r.t. q
  :param p: 4 x 1
  :return: 4 x 4
  """
  J = np.zeros((4, 4))
  J[0, 0]   = p[0]
  J[0, 1:]  = -p[1:].squeeze()
  J[1:, 0]  = p[1:].squeeze()
  J[1:, 1:] = p[0] * np.eye(3) + skew(p[1:])
  return J

def dpsq_q(p):
  """
  returns the jacobian of quaternion product (p*)q w.r.t. q
  :param p: 4 x 1
  :return: 4 x 4
  """
  J = np.zeros((4, 4))
  J[0, 0]   = p[0]
  J[0, 1:]  = -p[1:].squeeze()
  J[1:, 0]  = -p[1:].squeeze()
  J[1:, 1:] = p[0] * np.eye(3) - skew(p[1:])
  return J

def dpsq_p(q):
  """
  returns the jacobian of quaternion product (p*)q w.r.t. p
  :param q: 4 x 1
  :return: 4 x 4
  """
  J = np.zeros((4, 4))
  J[0, 0]   = q[0]
  J[0, 1:]  = q[1:].squeeze()
  J[1:, 0]  = q[1:].squeeze()
  J[1:, 1:] = -q[0] * np.eye(3) + skew(q[1:])
  return J

def dqstq_q(q, t):
  """
  jacobian of q* t q w.r.t. q
  :param q: 4 x 1
  :param t: 3 x 1
  :return: 3 x 4
  """
  J = np.zeros((3, 4))
  J[:, :1] = q[0]*t - np.cross(q[1:], t, axis=0)
  J[:, 1:] = -np.dot(t, q[1:].T) + np.dot(t.T, q[1:])*np.eye(3) + \
             np.dot(q[1:], t.T) + q[0]*skew(t)
  J *= 2
  return J

def dqstq_t(q):
  """
  jacobian of q* t q w.r.t. t
  :param q: 4 x 1
  :return: 3 x 3
  """
  J = (q[0]*q[0] - np.dot(q[1:].T, q[1:])) * np.eye(3) + 2*np.dot(q[1:], q[1:].T) -\
      2*q[0]*skew(q[1:])
  return J

def m_rot(x):
  """
  returns Jacobian of exponential map w.r.t. manifold increment
  :param x: part of state vector affected by increment, 4 x 1
  :return: 4 x 3
  """
  # jacobian of full q wrt qm (quaternion update on manifold),
  # evaluated at qv = (0, 0, 0)
  # full q is derived using either the exponential map or q0 = sqrt(1-qm^2)
  jm = np.vstack((np.zeros((1, 3)), np.eye(3)))  # 4 x 3
  m = np.dot(dpq_q(p=x), jm)
  return m

class PoseGraph:
  def __init__(self):
    """
    implements pose graph optimization from
    "Hybrid Hessians for Optimization of Pose Graphs" - Y. LeCun et al
    and "A Tutorial on Graph-Based SLAM" - W. Burgard et al
    """
    self.N = 0
    self.z = np.zeros((0, 0))

  def jacobian(self, L_ax, L_aq, L_rx, L_rq):
    J = np.zeros((0, 6*self.N))  # 6 because updates for rotation are on manifold

    # unary constraints
    for i in xrange(self.N):
      # translation constraint
      jt = np.zeros((3, J.shape[1]))
      jt[:, 6*i : 6*i+3] = np.eye(3)
      J = np.vstack((J, np.dot(L_ax, jt)))

      # rotation constraint
      jr = np.zeros((4, J.shape[1]))
      jr[:, 6*i+3 : 6*i+6] = m_rot(x=self.z[7*i+3 : 7*i+7])
      J = np.vstack((J, np.dot(L_aq, jr)))

    # pairwise constraints
    for i in xrange(self.N-1):
        # translation constraint
        jt = np.zeros((3, J.shape[1]))
        dt = dqstq_t(q=self.z[7*i+3 : 7*i+7])
        # dt = np.eye(3)
        jt[:, 6*i : 6*i+3] = -dt
        jt[:, 6*(i+1) : 6*(i+1)+3] = dt
        # m = m_rot(x=self.z[7*i+3 : 7*i+7])
        # a = dqstq_q(q=self.z[7*i+3 : 7*i+7],
        #             t=self.z[7*(i+1) : 7*(i+1)+3]-self.z[7*i : 7*i+3])
        # jt[:, 6*i+3 : 6*i+6] = np.dot(a, m)
        J = np.vstack((J, np.dot(L_rx, jt)))

        # rotation constraint
        jr = np.zeros((4, J.shape[1]))
        m = m_rot(x=self.z[7*i+3 : 7*i+7])
        a = dpsq_p(q=self.z[7*(i+1)+3 : 7*(i+1)+7])
        jr[:, 6*i+3 : 6*i+6] = np.dot(a, m)
        m = m_rot(x=self.z[7*(i+1)+3 : 7*(i+1)+7])
        b = dpsq_q(p=self.z[7*i+3 : 7*i+7])
        jr[:, 6*(i+1)+3 : 6*(i+1)+6] = np.dot(b, m)
        J = np.vstack((J, np.dot(L_rq, jr)))

    return J

  def residuals(self, poses, vos, L_ax, L_aq, L_rx, L_rq):
    """
    computes the residuals
    :param poses: N x 7
    :param vos: (N-1) x 7
    :param L_ax: 3 x 3
    :param L_aq: 4 x 4
    :param L_rx: 3 x 3
    :param L_rq: 4 x 4
    :return:
    """
    r = np.zeros((0, 1))

    # unary residuals
    L = np.zeros((7, 7))
    L[:3, :3] = L_ax
    L[3:, 3:] = L_aq
    for i in xrange(self.N):
      rr = self.z[7*i : 7*(i+1)] - np.reshape(poses[i], (-1, 1))
      r = np.vstack((r, np.dot(L, rr)))

    # pairwise residuals
    for i in xrange(self.N-1):
        # translation residual
        v = self.z[7*(i+1):7*(i+1)+3, 0]-self.z[7*i:7*i+3, 0]
        q = txq.qinverse(self.z[7*i+3:7*i+7, 0])
        rt = txq.rotate_vector(v, q)
        rt = rt[:, np.newaxis] - vos[i, :3].reshape((-1, 1))
        #rt = self.z[7*(i+1) : 7*(i+1)+3] - self.z[7*i : 7*i+3] - \
        #     vos[i, :3].reshape((-1, 1))
        r = np.vstack((r, np.dot(L_rx, rt)))

        # rotation residual
        q0 = self.z[7*i+3 : 7*i+7].squeeze()
        q1 = self.z[7*(i+1)+3 : 7*(i+1)+7].squeeze()
        qvo = txq.qmult(txq.qinverse(q0), q1).reshape((-1, 1))
        rq = qvo - vos[i, 3:].reshape((-1, 1))
        r = np.vstack((r, np.dot(L_rq, rq)))

    return r

  def update_on_manifold(self, x):
    """
    Updates the state vector on manifold
    :param x: manifold increment, column vector
    :return:
    """
    for i in xrange(self.N):
      # update translation
      t = x[6*i : 6*i+3]
      self.z[7*i : 7*i+3] += t

      # update rotation
      qm = x[6*i+3 : 6*i+6]  # quaternion on the manifold
      dq = np.zeros(4)
      # method in Burgard paper
      # dq[1:] = qm.squeeze()
      # dq[0] = math.sqrt(1 - sum(np.square(qm)))  # incremental quaternion
      # method of exponential map
      n = np.linalg.norm(qm)
      dq[0]  = math.cos(n)
      dq[1:] = np.sinc(n/np.pi) * qm.squeeze()
      q = self.z[7*i+3 : 7*i+7].squeeze()
      q = txq.qmult(q, dq).reshape((-1, 1))
      self.z[7*i+3 : 7*i+7] = q

  def optimize(self, poses, vos, sax=1, saq=1, srx=1, srq=1, n_iters=10):
    """
    run PGO, with init = poses
    :param poses:
    :param vos:
    :param sax: sigma for absolute translation
    :param saq: sigma for absolute rotation
    :param srx: sigma for relative translation
    :param srq: sigma for relative rotation
    :param n_iters:
    :return:
    """
    self.N = len(poses)
    # init state vector with the predicted poses
    self.z = np.reshape(poses.copy(), (-1, 1))

    # construct the information matrices
    L_ax = np.linalg.cholesky(np.eye(3) / sax)
    L_aq = np.linalg.cholesky(np.eye(4) / saq)
    L_rx = np.linalg.cholesky(np.eye(3) / srx)
    L_rq = np.linalg.cholesky(np.eye(4) / srq)

    for n_iter in xrange(n_iters):
      J = self.jacobian(L_ax.T, L_aq.T, L_rx.T, L_rq.T)
      r = self.residuals(poses.copy(), vos.copy(), L_ax.T, L_aq.T, L_rx.T,
                         L_rq.T)
      H = np.dot(J.T, J)  # hessian
      b = np.dot(J.T, r)  # residuals

      # solve Hx = -b for x
      R = slin.cholesky(H)  # H = R' R
      y = slin.solve_triangular(R.T, -b)
      x = slin.solve_triangular(R, y)

      self.update_on_manifold(x)

    return self.z.reshape((-1, 7))

class PoseGraphFC:
  def __init__(self):
    """
    implements pose graph optimization from
    "Hybrid Hessians for Optimization of Pose Graphs" - Y. LeCun et al
    and "A Tutorial on Graph-Based SLAM" - W. Burgard et al
    fully connected version
    """
    self.N = 0
    self.z = np.zeros((0, 0))

  def jacobian(self, L_ax, L_aq, L_rx, L_rq):
    J = np.zeros((0, 6*self.N))  # 6 because updates for rotation are on manifold

    # unary constraints
    for i in xrange(self.N):
      # translation constraint
      jt = np.zeros((3, J.shape[1]))
      jt[:, 6*i : 6*i+3] = np.eye(3)
      J = np.vstack((J, np.dot(L_ax, jt)))

      # rotation constraint
      jr = np.zeros((4, J.shape[1]))
      jr[:, 6*i+3 : 6*i+6] = m_rot(x=self.z[7*i+3 : 7*i+7])
      J = np.vstack((J, np.dot(L_aq, jr)))

    # pairwise constraints
    for i in xrange(self.N):
      for j in xrange(i+1, self.N):
        # translation constraint
        jt = np.zeros((3, J.shape[1]))
        dt = dqstq_t(q=self.z[7*i+3 : 7*i+7])
        # dt = np.eye(3)
        jt[:, 6*i : 6*i+3] = -dt
        jt[:, 6*j : 6*j+3] = dt
        # m = m_rot(x=self.z[7*i+3 : 7*i+7])
        # a = dqstq_q(q=self.z[7*i+3 : 7*i+7],
        #             t=self.z[7*(i+1) : 7*(i+1)+3]-self.z[7*i : 7*i+3])
        # jt[:, 6*i+3 : 6*i+6] = np.dot(a, m)
        J = np.vstack((J, np.dot(L_rx, jt)))

        # rotation constraint
        jr = np.zeros((4, J.shape[1]))
        m = m_rot(x=self.z[7*i+3 : 7*i+7])
        a = dpsq_p(q=self.z[7*j+3 : 7*j+7])
        jr[:, 6*i+3 : 6*i+6] = np.dot(a, m)
        m = m_rot(x=self.z[7*j+3 : 7*j+7])
        b = dpsq_q(p=self.z[7*i+3 : 7*i+7])
        jr[:, 6*j+3 : 6*j+6] = np.dot(b, m)
        J = np.vstack((J, np.dot(L_rq, jr)))

    return J

  def residuals(self, poses, vos, L_ax, L_aq, L_rx, L_rq):
    """
    computes the residuals
    :param poses: N x 7
    :param vos: (N-1) x 7
    :param L_ax: 3 x 3
    :param L_aq: 4 x 4
    :param L_rx: 3 x 3
    :param L_rq: 4 x 4
    :return: 
    """
    r = np.zeros((0, 1))

    # unary residuals
    L = np.zeros((7, 7))
    L[:3, :3] = L_ax
    L[3:, 3:] = L_aq
    for i in xrange(self.N):
      rr = self.z[7*i : 7*(i+1)] - np.reshape(poses[i], (-1, 1))
      r = np.vstack((r, np.dot(L, rr)))

    # pairwise residuals
    k = 0
    for i in xrange(self.N):
      for j in xrange(i+1, self.N):
        # translation residual
        v = self.z[7*j:7*j+3, 0]-self.z[7*i:7*i+3, 0]
        q = txq.qinverse(self.z[7*i+3:7*i+7, 0])
        rt = txq.rotate_vector(v, q)
        rt = rt[:, np.newaxis] - vos[k, :3].reshape((-1, 1))
        #rt = self.z[7*(i+1) : 7*(i+1)+3] - self.z[7*i : 7*i+3] - \
        #     vos[i, :3].reshape((-1, 1))
        r = np.vstack((r, np.dot(L_rx, rt)))

        # rotation residual
        q0 = self.z[7*i+3 : 7*i+7].squeeze()
        q1 = self.z[7*j+3 : 7*j+7].squeeze()
        qvo = txq.qmult(txq.qinverse(q0), q1).reshape((-1, 1))
        rq = qvo - vos[k, 3:].reshape((-1, 1))
        r = np.vstack((r, np.dot(L_rq, rq)))
        k += 1

    return r

  def update_on_manifold(self, x):
    """
    Updates the state vector on manifold
    :param x: manifold increment, column vector
    :return: 
    """
    for i in xrange(self.N):
      # update translation
      t = x[6*i : 6*i+3]
      self.z[7*i : 7*i+3] += t

      # update rotation
      qm = x[6*i+3 : 6*i+6]  # quaternion on the manifold
      dq = np.zeros(4)
      # method in Burgard paper
      # dq[1:] = qm.squeeze()
      # dq[0] = math.sqrt(1 - sum(np.square(qm)))  # incremental quaternion
      # method of exponential map
      n = np.linalg.norm(qm)
      dq[0]  = math.cos(n)
      dq[1:] = np.sinc(n/np.pi) * qm.squeeze()
      q = self.z[7*i+3 : 7*i+7].squeeze()
      q = txq.qmult(q, dq).reshape((-1, 1))
      self.z[7*i+3 : 7*i+7] = q

  def optimize(self, poses, vos, sax=1, saq=1, srx=1, srq=1, n_iters=10):
    """
    run PGO, with init = poses
    :param poses:
    :param vos:
    :param sax: sigma for absolute translation
    :param saq: sigma for absolute rotation
    :param srx: sigma for relative translation
    :param srq: sigma for relative rotation
    :param n_iters:
    :return:
    """
    self.N = len(poses)
    # init state vector with the predicted poses
    self.z = np.reshape(poses.copy(), (-1, 1))

    # construct the information matrices
    L_ax = np.linalg.cholesky(np.eye(3) / sax)
    L_aq = np.linalg.cholesky(np.eye(4) / saq)
    L_rx = np.linalg.cholesky(np.eye(3) / srx)
    L_rq = np.linalg.cholesky(np.eye(4) / srq)

    for n_iter in xrange(n_iters):
      J = self.jacobian(L_ax.T, L_aq.T, L_rx.T, L_rq.T)
      r = self.residuals(poses.copy(), vos.copy(), L_ax.T, L_aq.T, L_rx.T,
                         L_rq.T)
      H = np.dot(J.T, J)  # hessian
      b = np.dot(J.T, r)  # residuals

      # solve Hx = -b for x
      R = slin.cholesky(H)  # H = R' R
      y = slin.solve_triangular(R.T, -b)
      x = slin.solve_triangular(R, y)

      self.update_on_manifold(x)

    return self.z.reshape((-1, 7))

def optimize_poses(pred_poses, vos=None, fc_vos=False, target_poses=None,
                   sax=1, saq=1, srx=1, srq=1):
  """
  optimizes poses using either the VOs or the target poses (calculates VOs
  from them)
  :param pred_poses: N x 7
  :param vos: (N-1) x 7
  :param fc_vos: whether to use relative transforms between all frames in a fully
  connected manner, not just consecutive frames
  :param target_poses: N x 7
  :param: sax: covariance of pose translation (1 number)
  :param: saq: covariance of pose rotation (1 number)
  :param: srx: covariance of VO translation (1 number)
  :param: srq: covariance of VO rotation (1 number)
  :return:
  """
  pgo = PoseGraphFC() if fc_vos else PoseGraph()
  if vos is None:
    if target_poses is not None:
      # calculate the VOs (in the pred_poses frame)
      vos = np.zeros((len(target_poses)-1, 7))
      for i in xrange(len(vos)):
        vos[i, :3] = target_poses[i+1, :3] - target_poses[i, :3]
        q0 = target_poses[i, 3:]
        q1 = target_poses[i+1, 3:]
        vos[i, 3:] = txq.qmult(txq.qinverse(q0), q1)
    else:
      print 'Specify either VO or target poses'
      return None
  optim_poses = pgo.optimize(poses=pred_poses, vos=vos, sax=sax, saq=saq,
                             srx=srx, srq=srq)
  return optim_poses

def align_3d_pts(x1,x2):
  """Align two sets of 3d points using the method of Horn (closed-form).

  Find optimal s, R, t, such that

          s*R*(x1-t) = x2

  Input:
  x1 -- first trajectory (3xn)
  x2 -- second trajectory (3xn)

  Output:
  R -- rotation matrix (3x3)
  t -- translation vector (3x1)
  s -- scale (1x1)
  written by Jinwei Gu
  """
  x1c = x1.mean(1,keepdims=True)
  x2c = x2.mean(1,keepdims=True)

  x1_zerocentered = x1 - x1c
  x2_zerocentered = x2 - x2c

  W = np.zeros( (3,3) )
  r1 = 0
  r2 = 0
  for i in range(x1.shape[1]):
    a = x1_zerocentered[:,i]
    b = x2_zerocentered[:,i]
    W += np.outer(b, a)
    r1 += np.dot(a.T, a)
    r2 += np.dot(b.T, b)

  s = np.asscalar(np.sqrt(r2/r1))

  U,d,Vh = np.linalg.svd(W)
  S = np.eye(3)
  if np.linalg.det(np.dot(U, Vh))<0:
    S[2,2] = -1
  R = np.dot(U, np.dot(S, Vh))
  t = x1c - (1/s) * np.dot(R.transpose(), x2c)

  #---- align ----
  #x2a = s * np.dot(R, x1-t)
  #error = x2a - x2

  return R,t,s

def align_2d_pts(x1,x2):
  """Align two sets of 3d points using the method of Horn (closed-form).

  Find optimal s, R, t, such that

          s*R*(x1-t) = x2

  Input:
  x1 -- first trajectory (2xn)
  x2 -- second trajectory (2xn)

  Output:
  R -- rotation matrix (2x2)
  t -- translation vector (2x1)
  s -- scale (1x1)
  written by Jinwei Gu
  """
  x1c = x1.mean(1,keepdims=True)
  x2c = x2.mean(1,keepdims=True)

  x1_zerocentered = x1 - x1c
  x2_zerocentered = x2 - x2c

  W = np.zeros( (2,2) )
  r1 = 0
  r2 = 0
  for i in range(x1.shape[1]):
    a = x1_zerocentered[:,i]
    b = x2_zerocentered[:,i]
    W += np.outer(b, a)
    r1 += np.dot(a.T, a)
    r2 += np.dot(b.T, b)

  s = np.asscalar(np.sqrt(r2/r1))

  U,d,Vh = np.linalg.svd(W)
  S = np.eye(2)
  if np.linalg.det(np.dot(U, Vh))<0:
    S[1,1] = -1
  R = np.dot(U, np.dot(S, Vh))
  t = x1c - (1/s) * np.dot(R.transpose(), x2c)

  #---- align ----
  #x2a = s * np.dot(R, x1-t)
  #error = x2a - x2

  return R,t,s

def align_3d_pts_noscale(x1,x2):
  """Align two sets of 3d points using the method of Horn (closed-form).

  Find optimal s, R, t, such that

          s*R*(x1-t) = x2

  Input:
  x1 -- first trajectory (3xn)
  x2 -- second trajectory (3xn)

  Output:
  R -- rotation matrix (3x3)
  t -- translation vector (3x1)
  written by Jinwei Gu
  """
  x1c = x1.mean(1,keepdims=True)
  x2c = x2.mean(1,keepdims=True)

  x1_zerocentered = x1 - x1c
  x2_zerocentered = x2 - x2c

  W = np.zeros( (3,3) )
  r1 = 0
  r2 = 0
  for i in range(x1.shape[1]):
    a = x1_zerocentered[:,i]
    b = x2_zerocentered[:,i]
    W += np.outer(b, a)
    r1 += np.dot(a.T, a)
    r2 += np.dot(b.T, b)

  #s = np.asscalar(np.sqrt(r2/r1))
  s=1

  U,d,Vh = np.linalg.svd(W)
  S = np.eye(3)
  if np.linalg.det(np.dot(U, Vh))<0:
    S[2,2] = -1
  R = np.dot(U, np.dot(S, Vh))
  t = x1c - np.dot(R.transpose(), x2c)

  #---- align ----
  #x2a = s * np.dot(R, x1-t)
  #error = x2a - x2

  return R,t,s

def align_2d_pts_noscale(x1,x2):
  """Align two sets of 3d points using the method of Horn (closed-form).

  Find optimal s, R, t, such that

          s*R*(x1-t) = x2

  Input:
  x1 -- first trajectory (2xn)
  x2 -- second trajectory (2xn)

  Output:
  R -- rotation matrix (2x2)
  t -- translation vector (2x1)
  s -- scale (1x1)
  written by Jinwei Gu
  """
  x1c = x1.mean(1,keepdims=True)
  x2c = x2.mean(1,keepdims=True)

  x1_zerocentered = x1 - x1c
  x2_zerocentered = x2 - x2c

  W = np.zeros( (2,2) )
  r1 = 0
  r2 = 0
  for i in range(x1.shape[1]):
    a = x1_zerocentered[:,i]
    b = x2_zerocentered[:,i]
    W += np.outer(b, a)
    r1 += np.dot(a.T, a)
    r2 += np.dot(b.T, b)

  #s = np.asscalar(np.sqrt(r2/r1))
  s=1

  U,d,Vh = np.linalg.svd(W)
  S = np.eye(2)
  if np.linalg.det(np.dot(U, Vh))<0:
    S[1,1] = -1
  R = np.dot(U, np.dot(S, Vh))
  t = x1c - (1/s) * np.dot(R.transpose(), x2c)

  #---- align ----
  #x2a = s * np.dot(R, x1-t)
  #error = x2a - x2

  return R,t,s

def align_camera_poses(o1, o2, R1, R2, use_rotation_constraint=True):
  """Align two sets of camera poses (R1,o1/R2,o2) using the method of Horn (closed-form).

  Find optimal s, R, t, such that

          s*R*(o1-t) = o2   (1)

          R*R1 = R2         (2)

  where R1/R2 are the camera-to-world matrices, o1/o2 are the center
  of the cameras.

  Input:
  o1 -- camera centers (3xn)
  o2 -- camera centers (3xn)
  R1 -- camera poses (camera-to-world matrices) (nx3x3)
  R2 -- camera poses (camera-to-world matrices) (nx3x3)
  use_rotation_constraint -- if False, uses only Eq(1) to solve.

  Output:
  R -- rotation matrix (3x3)
  t -- translation vector (3x1)
  s -- scale (1x1)

  Note, when use_rotation_constraint=False, it is the same problem as
  above, i.e., to align two sets of 3D points.

  When use_rotation_constraint=True, we note Eq(2) is the same
  equation as Eq(1), after we zero-center and remove the scale. So, we
  can use the same approach (SVD).
  written by Jinwei Gu
  """
  if not use_rotation_constraint:
    return align_3d_pts(o1, o2)

  o1c = o1.mean(1,keepdims=True)
  o2c = o2.mean(1,keepdims=True)
  o1_zerocentered = o1 - o1c
  o2_zerocentered = o2 - o2c

  W = np.zeros( (3,3) )
  r1 = 0
  r2 = 0
  for i in range(o1.shape[1]):
    a = o1_zerocentered[:,i]
    b = o2_zerocentered[:,i]
    W += np.outer(b, a)
    r1 += np.dot(a.T, a)
    r2 += np.dot(b.T, b)

  s = np.asscalar(np.sqrt(r2/r1))

  # add rotation constraints
  for i in range(o1.shape[1]):
    d1 = np.squeeze(R1[i,:,:])
    d2 = np.squeeze(R2[i,:,:])
    for c in range(3):
      a = d1[:,c]
      b = d2[:,c]
      W += np.outer(b,a)

  U,d,Vh = np.linalg.svd(W)
  S = np.eye(3)
  if np.linalg.det(np.dot(U,Vh))<0:
    S[2,2] = -1
  R = np.dot(U, np.dot(S, Vh))
  t = o1c - (1/s) * np.dot(R.transpose(), o2c)

  #---- align ----
  #o2a = s * np.dot(R, o1-t)
  #R2a = np.dot(R, R1)

  return R,t,s

def test_align_3d_pts():
  import transforms3d.euler as txe
  N = 10
  x1 = np.random.rand(3,N)

  noise = np.random.rand(3,N)*0.01

  s = np.random.rand()
  t = np.random.rand(3,1)
  R = txe.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
  R = R[:3,:3]

  x2 = s*np.dot(R, x1-t) + noise

  Re,te,se = align_3d_pts(x1,x2)

  print 'scale ', s, se
  print 'rotation matrx ', R, Re
  print 'translation ', t, te

def test_align_camera_poses():
    import transforms3d.euler as txe

    N = 10
    o1 = np.random.rand(3,N)

    noise = np.random.rand(3,N)*0.01

    s = np.random.rand()
    t = np.random.rand(3,1)
    R = txe.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
    R = R[:3,:3]

    o2 = s*np.dot(R, o1-t) + noise

    R1 = np.zeros((N,3,3))
    R2 = np.zeros((N,3,3))
    for i in range(N):
      Ri = txe.euler2mat(np.random.rand(), np.random.rand(), np.random.rand())
      R1[i,:,:] = Ri[:3,:3]
      R2[i,:,:] = np.dot(R, Ri[:3,:3])

    Re1,te1,se1 = align_camera_poses(o1,o2,R1,R2,False)
    Re2,te2,se2 = align_camera_poses(o1,o2,R1,R2,True)

    print 'scale ', s, se1, se2
    print 'rotation matrx ', R, Re1, Re2
    print 'translation ', t, te1, te2

def pgo_test_poses():
  """
  generates test poses and vos for the various PGO implementations
  :return:
  """
  poses = np.zeros((3, 7))
  for i in xrange(poses.shape[0]):
    poses[i, :3] = i
    angle = math.radians(10*i)
    R = txe.euler2mat(angle, angle, angle)
    q = txq.mat2quat(R)
    poses[i, 3:] = q

  vos = np.zeros((poses.shape[0]-1, 7))
  for i in xrange(vos.shape[0]):
    vos[i, 0] = 1.5
    vos[i, 1] = 0.5
    vos[i, 2] = 1.0
    R = txe.euler2mat(math.radians(15), math.radians(10), math.radians(5))
    q = txq.mat2quat(R)
    vos[i, 3:] = q

  return poses, vos

def pgo_test_poses1():
  poses = np.zeros((3, 7))
  R = txe.euler2mat(0, 0, np.deg2rad(45))
  q = txq.mat2quat(R)
  poses[:, 3:] = q
  for i in xrange(len(poses)):
    poses[i, :3] = np.asarray([i, i, 0])

  pt = np.zeros((len(poses), 6))
  pt[:, :3] = poses[:, :3]
  for i,p in enumerate(poses):
    pt[i, 3:] = qlog(p[3:])
  pt = torch.from_numpy(pt.astype(np.float32))
  vost = calc_vos_safe_fc(pt.unsqueeze(0))[0].numpy()
  vos = np.zeros((len(vost), 7))
  vos[:, :3] = vost[:, :3]
  for i,p in enumerate(vost):
    vos[i, 3:] = qexp(p[3:])

  # perturbation
  vos[0, 0]  = np.sqrt(2) - 0.5
  vos[1, 0]  = np.sqrt(2) - 0.5

  return poses, vos

def print_poses(poses):
  print 'translations'
  print poses[:, :3]
  print 'euler'
  for i in xrange(poses.shape[0]):
    a = txe.mat2euler(txq.quat2mat(poses[i, 3:]))
    print [np.rad2deg(aa) for aa in a]

def test_pgo():
  """
  Tests the full pose graph optimization implementation
  :return: bool
  """
  pred_poses, vos = pgo_test_poses1()
  print 'pred poses'
  print_poses(pred_poses)
  print 'vos'
  print_poses(vos)

  pgo = PoseGraph()
  optimized_poses = pgo.optimize(pred_poses, vos)

  print 'optimized'
  print_poses(optimized_poses)


def test_pose_utils():
  """
  Tests the pose utils
  :return: 
  """
  TEST_COMPOSE = True
  TEST_INV = True

  ra = lambda _: np.random.uniform(0, 2*math.pi)

  if TEST_COMPOSE:
    print 'Testing pose composing...'
    R1 = txe.euler2mat(ra(1), ra(1), ra(1))
    t1 = np.random.rand(3)
    R2 = txe.euler2mat(ra(1), ra(1), ra(1))
    t2 = np.random.rand(3)

    # homogeneous matrix method
    R = np.dot(R1, R2)
    t = t1 + np.dot(R1, t2)
    print 'From homogeneous matrices, t = '
    print t
    print 'R = '
    print R

    # quaternion method
    q1 = txq.mat2quat(R1)
    q2 = txq.mat2quat(R2)

    p1 = torch.cat((torch.from_numpy(t1), torch.from_numpy(q1)))
    p2 = torch.cat((torch.from_numpy(t2), torch.from_numpy(q2)))
    p  = compose_pose_quaternion(torch.unsqueeze(p1, 0), torch.unsqueeze(p2, 0))
    t  = p[:, :3].numpy().squeeze()
    q  = p[:, 3:].numpy().squeeze()
    print 'From quaternions, t = '
    print t
    print 'R = '
    print txe.quat2mat(q)

  if TEST_INV:
    print 'Testing pose inversion...'
    R = txe.euler2mat(ra(1), ra(1), ra(1))
    t = np.random.rand(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, -1] = t

    q = txq.mat2quat(R)
    p = torch.cat((torch.from_numpy(t), torch.from_numpy(q)))
    pinv = invert_pose_quaternion(torch.unsqueeze(p, 0))
    tinv, qinv = pinv[:, :3], pinv[:, 3:]
    Rinv = txq.quat2mat(qinv.numpy().squeeze())
    Tinv = np.eye(4)
    Tinv[:3, :3] = Rinv
    Tinv[:3, -1] = tinv.numpy().squeeze()
    print 'T * T^(-1) = '
    print np.dot(T, Tinv)

def test_q_error():
  ra = lambda _: np.random.uniform(0, 2*math.pi)
  # rotation along x axis
  a1 = ra(1)
  a2 = ra(1)
  q1 = txq.mat2quat(txe.euler2mat(a1, 0, 0))
  q2 = txq.mat2quat(txe.euler2mat(a2, 0, 0))
  a1 = np.rad2deg(a1)
  a2 = np.rad2deg(a2)
  print 'Angles: {:f}, {:f}, difference = {:f}'.format(a1, a2, a1-a2)
  print 'Error: {:f}'.format(quaternion_angular_error(q1, q2))

def test_log_q_error():
  ra = lambda _: np.random.uniform(0, 2*math.pi)
  # rotation along x axis
  a1 = ra(1)
  a2 = ra(1)
  q1 = txq.mat2quat(txe.euler2mat(0, a1, 0))
  q2 = txq.mat2quat(txe.euler2mat(0, a2, 0))
  # apply log map
  q1 = np.arccos(q1[0]) * q1[1:] / np.linalg.norm(q1[1:])
  q2 = np.arccos(q2[0]) * q2[1:] / np.linalg.norm(q2[1:])
  a1 = np.rad2deg(a1)
  a2 = np.rad2deg(a2)
  print 'Angles: {:f}, {:f}, difference = {:f}'.format(a1, a2, a1-a2)
  print 'Error: {:f}'.format(log_quaternion_angular_error(q1, q2))

if __name__ == '__main__':
  test_pgo()
  # test_dumb_pgo()
  # test_align_camera_poses()
  # test_q_error()
  # test_log_q_error()
