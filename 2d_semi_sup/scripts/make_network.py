
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
from network_defs import *
import numpy as np
import h5py


# In[2]:

batch_size=4
n = caffe.NetSpec()
n.data = L.NetCDFData(source="/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/source_files.txt",
                   variable_data=net_vars, time_stride=2,
                    batch_size=batch_size, name="foo")

n.label = L.HDF5Data(source="/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/label_files.txt",
                     batch_size=batch_size)

n.gxy, n.gwh, n.gobj, n.gcls = L.Slice(n.label, slice_point=[2,4,5], ntop=4)


pstr, fn = write_to_file(n, "filt_test")


# In[3]:

print pstr


# In[4]:

#pstr, fn, n= make_netcdf_network()


# In[5]:

if __name__ == "__main__":
    #fn = "/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/sm_sq_2d_16.prototxt"
    net=caffe.Net(fn,caffe.TRAIN)

    a =net.forward()

    net.backward()


# In[6]:

a["gcls"][0]


# In[20]:

lbl =net.blobs["label"].data


# In[21]:

b =net.blobs['data'].data


# In[22]:

b.shape


# In[8]:

from matplotlib import pyplot as plt


# In[9]:

get_ipython().magic(u'matplotlib inline')


# In[10]:

plt.imshow(b[0][6])


# In[7]:

import h5py


# In[8]:

hf = h5py.File("/global/cscratch1/sd/racah/TCHero/labels/cam5_1_amip_run2.cam2.h2.2005-12-31-10800.h5")


# In[11]:

hf["label"][0,-1]


# In[ ]:



