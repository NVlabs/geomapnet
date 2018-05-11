"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_batch(batch):
  npimg = batch.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  plt.show()

def show_stereo_batch(l_batch, r_batch):
  l_npimg = np.transpose(l_batch.numpy(), (1,2,0))
  r_npimg = np.transpose(r_batch.numpy(), (1,2,0))
  plt.imshow(np.concatenate((l_npimg, r_npimg), axis=1), interpolation='nearest')
  plt.show()

def vis_tsne(embedding, images, ax=None):
  """

  :param embedding:
  :param images: list of PIL images
  :param ax:
  :return:
  """
  if ax is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

  embedding -= np.min(embedding, axis=0)
  embedding /= np.max(embedding, axis=0)
  S = 5000
  s = 250
  canvas = np.zeros((S, S, 3), dtype=np.uint8)
  for pos, im in zip(embedding[:, [1, 2]], images):
    x, y = (pos * S).astype(np.int)
    im.thumbnail((s, s), Image.ANTIALIAS)
    x = min(x, S-im.size[0])
    y = min(y, S-im.size[1])
    canvas[y:y+im.size[1], x:x+im.size[0], :] = np.array(im)

  ax.imshow(canvas)
  plt.show(block=True)

if __name__ == '__main__':
  import cPickle
  with open('../data/embedding_data_robotcar_vidvoo.pkl', 'rb') as f:
    print 'Reading data...'
    embedding, images = cPickle.load(f)
    print 'done'

  vis_tsne(embedding, images)
