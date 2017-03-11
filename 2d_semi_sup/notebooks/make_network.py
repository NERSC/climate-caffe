
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from make_solver import make_solver
import argparse
from os.path import join


# In[2]:

#%matplotlib inline


# In[3]:

cl_args = {"lr": 0.01,
           "num_epochs": 20, 
           "filters_scale": 1./8,
           "data_dir": "smallish_dataset", 
           "save_dir": "/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/notebooks/plots" }


# In[4]:

if any(["jupyter" in arg for arg in sys.argv]):
    sys.argv=sys.argv[:1]
    
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
for k,v in cl_args.iteritems():
    parser.add_argument('--' + k, type=type(v), default=v, help=k)

args = parser.parse_args()
cl_args.update(args.__dict__)


# In[5]:

basepath = "/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/notebooks/proto_files/"

tr_batch_size=4
val_batch_size=16


# In[6]:

tr_pstr, tr_netspec, num_tr = make_netcdf_network(batch_size=tr_batch_size,
                                                  prefix_dir=cl_args["data_dir"],
                                                  mode="tr", 
                                                  filters_scale=cl_args["filters_scale"] )



_, tr_net_filepath = write_to_file(tr_netspec, filename="train", basepath=basepath)

val_pstr, val_netspec, num_val = make_netcdf_network(batch_size=val_batch_size,prefix_dir=cl_args["data_dir"], mode="val" ,filters_scale=cl_args["filters_scale"])
_, val_net_filepath = write_to_file(val_netspec, filename="val", basepath=basepath, )


# In[7]:

spstr,solver_filename = make_solver(net_path=basepath,base_lr=cl_args["lr"],
                                    train_file_name="train.prototxt", 
                                    test_net_path=val_net_filepath, tr_num_examples=num_tr, 
                                    test_num_examples=num_val,tr_batch_size=tr_batch_size, 
                                    test_batch_size=val_batch_size, print_every_iteration=10000)


# In[8]:




# In[9]:

solver = caffe.SGDSolver(solver_filename)


# In[ ]:

tr_its_per_epoch = int(float(num_tr) / tr_batch_size)
val_its_per_epoch = int(float(num_val) / val_batch_size)
tr_ep_losses = []
test_ep_losses = []
it_losses = []
# cls_losses = []
# rec_losses = []
num_epochs = cl_args["num_epochs"]
for ep in range(num_epochs):
    for _ in range(tr_its_per_epoch):
        solver.step(1)

        loss = np.float32(solver.net.blobs['final_loss'].data)
        #rec_losses.append(np.float32(solver.net.blobs['L_rec'].data))
        #cls_losses.append(np.float32(solver.net.blobs['L_cls'].data))
        it_losses.append(loss)
    

    #sys.stderr.write("TR CLS LOSS: " + str(np.mean(cls_losses)))
    #print "TR REC LOSS: ", np.mean(rec_losses)
    ep_loss = np.mean(it_losses)
    
    sys.stderr.write("\nEPOCH TR: " + str(ep_loss))
    tr_ep_losses.append(ep_loss)
    it_losses = []
#     cls_losses = []
#     rec_losses = []
    
    
#     for _ in range(val_its_per_epoch):
#         tn=solver.test_nets[0]
#         loss = np.float32(tn.forward()["final_loss"].data)
#         it_losses.append(loss)
    
#     test_ep_loss = np.mean(it_losses)
#     test_ep_losses.append(test_ep_loss)
#     sys.stderr.write("\nEPOCH VAL: " + str(test_ep_loss))
#     it_losses = []
    
    
    suffix = "learning_curve_lr_%4.2f_fs_%4.3f_%s.jpg"%(cl_args["lr"], cl_args["filters_scale"], cl_args["data_dir"])
    plt.figure(1)
    plt.plot(tr_ep_losses)
    plt.savefig(join(cl_args["save_dir"],"tr_" + suffix))
#     plt.figure(2)
#     plt.plot(test_ep_losses)
#     plt.savefig(join(cl_args["save_dir"],"val_" + suffix))
   
    


# In[8]:

#! jupyter nbconvert --to script make_network.ipynb


# In[15]:

# gb_basedir = "/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup"

# val_net_filepath = join(gb_basedir, "val.prototxt")

# spstr,solver_filename = make_solver(net_path=gb_basedir,base_lr=cl_args["lr"],
#                                     train_file_name="train.prototxt", 
#                                     test_net_path=val_net_filepath, tr_num_examples=num_tr, 
#                                     test_num_examples=num_val,tr_batch_size=tr_batch_size, 
#                                     test_batch_size=val_batch_size, print_every_iteration=10000)


# In[ ]:



