
# coding: utf-8

# In[1]:

import caffe

from caffe import layers as L, params as P, to_proto

from caffe.proto import caffe_pb2

from caffe.coord_map import crop

import copy
from nbfinder import NotebookFinder
import sys
sys.meta_path.append(NotebookFinder())
from layer_util import *


CAFFE=1
SUM=1

def make_loss(n):
    n.loss1 = L.EuclideanLoss(n.yolo, n.label)
    n.lossr = L.EuclideanLoss(n.decoder,n.data)
    n.final_loss = L.Eltwise(n.loss1,n.lossr,operation=SUM)
    return n


# In[ ]:



