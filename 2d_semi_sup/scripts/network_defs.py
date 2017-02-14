
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
from network_architecture import *
from loss_acc import *


# In[3]:



net_vars = ["PRECT",
"PS",
"PSL",
"QREFHT",
"T200",
"T500",
"TMQ",
"TREFHT",
"TS",
"U850",
"UBOT",
"V850",
"VBOT",
"Z1000",
"Z200",
"ZBOT"]


# In[4]:

lbls = ['teca_mask',
 'teca_xmin',
 'teca_xmax',
 'teca_ymin',
 'teca_ymax',
 'teca_category']


# In[18]:


def make_netcdf_network(inp_x = 768,inp_y=1152,num_classes=4, name="netcdf"):
    num_channels = len(net_vars)
    n = caffe.NetSpec()
    n.data = L.NetCDFData(source="/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/source_files.txt",
                       variable_data=net_vars, time_stride=2,
                        batch_size=4, name="foo")
    
    n.label = L.HDF5Data(source="/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/label_files.txt",
                         batch_size=4)
    nfilters_list = [128, 256, 512, 768, 1024, 1280]
    n = create_net(n,n.data,nfilters_list, num_classes, num_channels)
    n = make_loss(n)

    pstr, fn = write_to_file(n, name)
    return pstr, fn, n
    

    



def make_dummy_network(inp_x = 768,inp_y=1152,lbl_ch = (4,4,2),batch_size=2, num_layers=6, name="sm_sq_2d"):
    nfilters_list = [128, 256, 512, 768, 1024, 1280]
    n = caffe.NetSpec()
    n.data = L.DummyData(shape={'dim':[batch_size,16,inp_x,inp_y]}, data_filler={"type":"gaussian", "mean":0, "std":1})
    lbl_x = inp_x / 64
    lbl_y = inp_y / 64
    n.label1 = L.DummyData(shape={'dim':[batch_size,lbl_ch[0],lbl_x, lbl_y]}, data_filler={"type":"gaussian", "mean":0, "std":1})
    n.label2 = L.DummyData(shape={'dim':[batch_size,lbl_ch[1],lbl_x, lbl_y]}, data_filler={"type":"gaussian", "mean":0, "std":1})
    n.label3 = L.DummyData(shape={'dim':[batch_size,lbl_ch[2],lbl_x, lbl_y]}, data_filler={"type":"gaussian", "mean":0, "std":1})
    n.label = L.Concat(n.label1, n.label2, n.label3)
    n = create_net(n,n.data,nfilters_list, lbl_ch, num_chanels=16)
    n = make_loss(n)



    pstr, fn = write_to_file(n, name)
    return pstr, fn, n



# In[6]:

n = caffe.NetSpec()
n.data = L.NetCDFData(source="/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/source_files.txt",
                       variable_data=net_vars, time_stride=2,
                        batch_size=4, name="foo")

