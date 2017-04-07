
import matplotlib; matplotlib.use("agg")


import sys
import numpy as np
import time
from label_loader import  make_labels_for_dataset
import h5py
import os
from os.path import join
from util import get_timestamp
import netCDF4 as nc
import argparse



def get_gr_truth_configs(kwargs):
    scale_factor = float(kwargs["scale_factor"])
    xdim, ydim = kwargs["xdim"], kwargs["ydim"]
    num_classes = kwargs["num_classes"]
    #make sure xy coords divide cleanly with scale_factor
    assert xdim % scale_factor == 0 and ydim % scale_factor == 0, "scale factor %i must divide the xy (%i, %i) coords cleanly " %(scale_factor,xdim, ydim)
    
    xlen, ylen = xdim / int(scale_factor), ydim / int(scale_factor)
    
    #x,y,w,h,conf1,conf2 plus num_classes for one hot encoding
    if kwargs["caffe_format"]:
        last_dim = 6  #(xywh obj cls)
    else:
        last_dim = 6 + num_classes
    return scale_factor, xlen, ylen, last_dim, num_classes 
    



def get_xy_inds(x,y, scale_factor):
        # get the indices to the lower left corner of the grid
        
        #scale x and y down
        xs, ys = x / scale_factor, y / scale_factor
        eps = 10*np.finfo(float).eps
        #take the floor of x and y -> which is rounding to nearest bottom left corner
        x_ind, y_ind = [np.floor(k - 10*eps ).astype('int') for k in [xs,ys]]
        return x_ind, y_ind
    



def get_xy_offsets(x,y, x_ind, y_ind, scale_factor):
    #scale x and y down
    xs, ys = x / scale_factor, y / scale_factor
    
    #get the offsets by subtracting the scaled lower left corner coords from the scaled real coords
    xoff, yoff = xs - x_ind, ys - y_ind
    
    return xoff, yoff



def get_parametrized_wh(w,h,scale_factor):
    ws , hs = w / scale_factor, h/ scale_factor
    wp, hp = np.log2(ws), np.log2(hs)
    return wp, hp
    



def convert_class_to_one_hot(class_num, num_classes):
    vec = num_classes * [0]
    vec[class_num - 1] = 1
    return vec



def get_box_vector(coords, scale_factor, num_classes, caffe_format):
    x,y,w,h,cls = coords
    xind, yind = get_xy_inds(x,y,scale_factor)
    xoff, yoff = get_xy_offsets(x, y, xind, yind, scale_factor)
    wp, hp = get_parametrized_wh(w, h, scale_factor)
    if caffe_format:
        cls_vec = [cls] #classes are 1-4 (no zero on purpose, so that can be filtered out)
        objectness_vec = [1]
    else:
        cls_vec = convert_class_to_one_hot(cls, num_classes=num_classes)
        objectness_vec = [1, 0]
    box_loc = [xoff, yoff, wp, hp]
    box_vec = np.asarray(box_loc + objectness_vec + cls_vec)
    return box_vec



def test_grid(bbox, grid,kwargs):
    xdim, ydim, scale_factor,num_classes, caffe_format = kwargs["xdim"], kwargs["ydim"],kwargs["scale_factor"], kwargs["num_classes"],kwargs["caffe_format"]
    scale_factor = float(scale_factor)
    cls = int(bbox[4])
    x,y = bbox[0] / scale_factor, bbox[1] / scale_factor

    xo,yo = x - np.floor(x), y - np.floor(y)
    w,h = np.log2(bbox[2] / scale_factor), np.log2(bbox[3] / scale_factor)



    if caffe_format:
        depth = 6
        caffe_box = grid[:depth,int(x),int(y)]
        l_box = caffe_box
        lbl = [cls]
        obj = [1.]
    else:
        depth = 6 + num_classes
        oth_box = grid[int(x),int(y),:depth]
        l_box = oth_box
        obj = [1., 0.]
        

        lbl = num_classes*[0]
        lbl[cls-1] = 1
    
    real_box = [xo,yo,w,h] + obj
    real_box.extend(lbl)
    
    print l_box

    print real_box
    assert np.allclose(l_box, real_box), "Tests Failed"
#     if np.allclose(l_box, real_box) == True:
#         print "Yay! Passed Test"



def make_default_no_object_1hot(gr_truth):
    #make the 5th number 1, so the objectness by defualt is 0,1 -> denoting no object
    gr_truth[:,:,:,5] = 1.
    return gr_truth



def create_yolo_gr_truth(bbox_list, kwargs, year):
        caffe_format = kwargs["caffe_format"]
        scale_factor, xlen, ylen, last_dim, num_classes = get_gr_truth_configs(kwargs)

        num_time_steps = len(bbox_list)
        
        gr_truth = np.zeros(( num_time_steps, xlen, ylen, last_dim ))

        # if the file is specified as unlabelled then we skip this
        # and return gr_truth as is -> all zeros
        if year not in kwargs["unlabelled_years"]:
            if not caffe_format:
                gr_truth = make_default_no_object_1hot(gr_truth)
            # for caffe we have the channels as the following x,y,w,h,obj,cls
            # obj is 1 or 0 and cls is 1-4 if an obj is there and 0 if not
            #For noncaffe we have x,y,w,h,obj,no-obj, cls1,cls2,cls3,cls4
            #cls1-cls4 is one hot encoded vector
            for time_step in range(num_time_steps):
                for coords in bbox_list[time_step]:
                    x,y,w,h,cls = coords

                    xind, yind = get_xy_inds(x,y,scale_factor)
                    box_vec = get_box_vector(coords, scale_factor, num_classes, caffe_format)
                    gr_truth[time_step,xind,yind,:] = box_vec

        if caffe_format:
            gr_truth = np.transpose(gr_truth, axes=(0,3,1,2))
        return gr_truth



def make_yolo_masks_for_dataset( camfile_path, kwargs, labels_csv_file):
        ts = get_timestamp(camfile_path)
        box_list = make_labels_for_dataset(camfile_path, labels_csv_file)
        yolo_mask = create_yolo_gr_truth(box_list, kwargs, ts.year)
        return yolo_mask



def save_mask(camfile_name, mask, save_loc, fmt="netcdf"):

    def save_mask_netcdf():
        netcdf_name = camfile_name.split(".nc")[0] + "_label.nc"
        rootgrp = nc.Dataset(join(save_loc, netcdf_name), "w")

        time = rootgrp.createDimension("time", mask.shape[0])
        xdim = rootgrp.createDimension("x", mask.shape[2])
        ydim = rootgrp.createDimension("y", mask.shape[3])

        x_coord = rootgrp.createVariable("x_coord","f4",("time","x","y"))
        y_coord = rootgrp.createVariable("y_coord","f4",("time","x","y"))
        w_coord = rootgrp.createVariable("w_coord","f4",("time","x","y"))
        h_coord = rootgrp.createVariable("h_coord","f4",("time","x","y"))
        obj = rootgrp.createVariable("obj","f4",("time","x","y"))
        cls = rootgrp.createVariable("cls","f4",("time","x","y"))

        vars_ = [x_coord, y_coord, w_coord, h_coord, obj, cls]
        for i,var in enumerate(vars_):
            # for every example for the correct variable, for al x and y
            var[:] = mask[:,i,:,:]

    
    
        rootgrp.close()
    
    def save_mask_hdf5():
        h5f_name = camfile_name.split(".nc")[0] + ".h5"
        hf = h5py.File(join(save_loc, h5f_name))
        hfd = hf.create_dataset(name="label", data=mask)
        hf.close()
    if fmt == "netcdf":
        save_mask_netcdf()
    else:
        save_mask_hdf5()




    



def save_label_tensors(kwargs):

    


    labels_csv_file = join(kwargs["metadata_dir"], "labels.csv")
    


    for camfile_name in os.listdir(kwargs["data_dir"]):

        camfile_path = join(kwargs["data_dir"], camfile_name)
        
        if "amip" in camfile_name:
            ym = make_yolo_masks_for_dataset(camfile_path,
                                     kwargs,
                                    labels_csv_file)
            if kwargs["with_unlabelled_frames"]:
                ym_z = np.zeros((kwargs["time_steps_per_file"], ym.shape[1], ym.shape[2], ym.shape[3]))
                ym_z[0::2] = ym
                ym = ym_z
        else:
            ym = np.zeros((kwargs["time_steps_per_file"], 
                           6, 
                           kwargs["xdim"] / kwargs["scale_factor"], 
                           kwargs["ydim"] / kwargs["scale_factor"]
                          ))


        #test(camfile_path, ym, kwargs)
        save_mask(camfile_name, ym, save_loc=kwargs["dest_dir"], fmt=kwargs["file_format"])


            
            
            
    
    



def test(camfile_path, mask, kwargs):
    labels_csv_file = join(kwargs["metadata_dir"], "labels.csv")
    box_list = make_labels_for_dataset(camfile_path, labels_csv_file)
    box = box_list[0][0]
    test_grid(box, mask[0], kwargs)
    



def test_hdf5_file(h5f_path, camfile_path, kwargs):


    labels_csv_file = join(kwargs["metadata_dir"], "labels.csv")
    ym = h5py.File(h5f_path)["label"]
    test(camfile_path, ym, kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dest_dir', type=str)

    args = parser.parse_args().__dict__
    
    kwargs = {  "metadata_dir": "/global/cscratch1/sd/racah/TCHero/labels",
                "data_dir": "/global/cscratch1/sd/racah/TCHero/data",
                "scale_factor": 32, 
                "xdim":768,
                "ydim":1152,
                "time_steps_per_file": 8,
                "file_format": "netcdf",
                "with_unlabelled_frames": True,
                "dest_dir": "/global/cscratch1/sd/racah/TCHero/labels_nc",
                "num_classes": 4, "caffe_format": True, "unlabelled_years": [] }
    kwargs.update(args)

    save_label_tensors(kwargs)
    



# import matplotlib.pyplot as plt

# %matplotlib inline

# d=nc.Dataset(
#     """/project/projectdirs/dasrepo/gordon_bell/deep_learning/data/climate/CAM5_0.25/2xCO2/labels/cam5_1_CLIVAR_2xCO2_quarterdegree_branch2.cam2.h2.0013-06-09-10800_label.nc""")

# plt.imshow(d["obj"][7])






