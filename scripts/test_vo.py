"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import numpy as np
import transforms3d.quaternions as txq
import transforms3d.euler as txe

# predicted poses
tp1 = np.random.rand(3)
tp2 = np.random.rand(3)
qp1 = txe.euler2quat(*(2 * np.pi * np.random.rand(3)))
qp2 = txe.euler2quat(*(2 * np.pi * np.random.rand(3)))

# relatives
t_rel = txq.rotate_vector(v=tp2-tp1, q=txq.qinverse(qp1))
q_rel = txq.qmult(txq.qinverse(qp1), qp2)

# vo poses
trand = np.random.rand(3)
qrand = txe.euler2quat(*(2 * np.pi * np.random.rand(3)))
tv1 = txq.rotate_vector(v=tp1, q=qrand)
qv1 = txq.qmult(qrand, qp1)
tv2 = txq.rotate_vector(v=t_rel, q=qv1) + tv1
qv2 = txq.qmult(qv1, q_rel)

# aligned vo
voq = txq.qmult(txq.qinverse(qv1), qv2)
vot = txq.rotate_vector(v=tv2-tv1, q=txq.qinverse(qv1))
vot = txq.rotate_vector(v=vot, q=qp1)

print 'translation'
print np.allclose(tp1 + vot, tp2)

print 'rotation'
print np.allclose(txq.qmult(qp1, voq), qp2)
