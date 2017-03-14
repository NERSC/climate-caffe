
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




# In[3]:

cl_args = {"lr": 0.00001,
           "num_epochs": 20, 
           "filters_scale": 1./8,
           "data_dir": "extremely_small_dataset", 
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

tr_pstr, tr_netspec, num_tr = make_netcdf_network(batch_size=tr_batch_size,
                                                  prefix_dir=cl_args["data_dir"],
                                                  mode="tr", 
                                                  filters_scale=cl_args["filters_scale"] )



_, tr_net_filepath = write_to_file(tr_netspec, filename="train", basepath=basepath)

val_pstr, val_netspec, num_val = make_netcdf_network(batch_size=val_batch_size,prefix_dir=cl_args["data_dir"], mode="val" ,filters_scale=cl_args["filters_scale"])
_, val_net_filepath = write_to_file(val_netspec, filename="val", basepath=basepath, )


# In[6]:

spstr,solver_filename = make_solver(net_path=basepath,base_lr=cl_args["lr"],
                                    train_file_name="train.prototxt", 
                                    test_net_path=val_net_filepath, tr_num_examples=num_tr, 
                                    test_num_examples=num_val,tr_batch_size=tr_batch_size, 
                                    test_batch_size=val_batch_size, print_every_iteration=10000)


# In[7]:

solver = caffe.SGDSolver(solver_filename)


# In[ ]:

#%matplotlib inline


# In[8]:

tr_its_per_epoch = int(float(num_tr) / tr_batch_size)
val_its_per_epoch = int(float(num_val) / val_batch_size)
num_epochs = cl_args["num_epochs"]

loss_keys = ["L_cls",
"L_obj",
"L_xy", 
"L_wh",
"L_rec",
"final_loss"]
losses = {}
for ep in range(num_epochs):
    losses[ep] = {k:[] for k in loss_keys}
    losses[str(ep) + "_mean"] = {k:0 for k in loss_keys}
    for it in range(tr_its_per_epoch):
        solver.step(1)

        for k in loss_keys:
            loss = np.float32(solver.net.blobs[k].data)
            losses[ep][k].append(loss)
            


    losses[str(ep) + "_mean"] = {k:np.mean(losses[ep][k]) for k in loss_keys}
    
    
    for k in loss_keys:
        loss = losses[str(ep) + "_mean"][k]
        sys.stderr.write("\n Epoch %i: Final %s Loss = %6.3f" % (ep, k, loss))
        
        
        

    
    
    suffix = "learning_curve_lr_%8.6f_fs_%4.3f_%s.jpg"%(cl_args["lr"], cl_args["filters_scale"], cl_args["data_dir"])
    plt.figure(1)
    plt.plot([losses[str(epoch) + "_mean"]["final_loss"] for epoch in range(ep + 1)])
    #plt.show()
    plt.savefig(join(cl_args["save_dir"],"tr_" + suffix))

   
    


# In[ ]:

#! jupyter nbconvert --to script make_network.ipynb


# In[ ]:




# In[11]:

# gb_basedir = "/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup"

# tr_batch_size=4
# val_batch_size=16

# tr_pstr, tr_netspec, num_tr = make_netcdf_network(batch_size=tr_batch_size,
#                                                   prefix_dir=cl_args["data_dir"],
#                                                   mode="tr", 
#                                                   filters_scale=cl_args["filters_scale"] )



# _, tr_net_filepath = write_to_file(tr_netspec, filename="train", basepath=gb_basedir)

# val_pstr, val_netspec, num_val = make_netcdf_network(batch_size=val_batch_size,prefix_dir=cl_args["data_dir"], mode="val" ,filters_scale=cl_args["filters_scale"])
# _, val_net_filepath = write_to_file(val_netspec, filename="val", basepath=gb_basedir, )

# val_net_filepath = join(gb_basedir, "val.prototxt")

# spstr,solver_filename = make_solver(net_path=gb_basedir,base_lr=cl_args["lr"],
#                                     train_file_name="train.prototxt", 
#                                     test_net_path=val_net_filepath, tr_num_examples=num_tr, 
#                                     test_num_examples=num_val,tr_batch_size=tr_batch_size, 
#                                     test_batch_size=val_batch_size, print_every_iteration=10000)

# solver = caffe.SGDSolver(solver_filename)

# solver.step(1)


# In[ ]:



