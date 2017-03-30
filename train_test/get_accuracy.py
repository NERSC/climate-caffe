
# coding: utf-8

# In[1]:

import caffe
import sys
import argparse
# from nbfinder import NotebookFinder
# import sys
# sys.meta_path.append(NotebookFinder())
from accuracy import BBox_Accuracy
import numpy as np


# In[16]:

if __name__ == "__main__":
    def get_cl_args():
        cl_args = {"path_to_caffemodel":"/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/models/_iter_10.caffemodel",
                    "path_to_caffe_prototxt":"/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/train_intel_goodperf_vanilla.prototxt",
                   "iterations": 3, "iou_thresh":0.5}

        if any(["jupyter" in arg for arg in sys.argv]):
            sys.argv=sys.argv[:1]

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for k,v in cl_args.iteritems():
            parser.add_argument('--' + k, type=type(v), default=v, help=k)

        args = parser.parse_args()
        cl_args.update(args.__dict__)
        return cl_args

    cl_args = get_cl_args()
    net = caffe.Net(cl_args["path_to_caffe_prototxt"],
                    cl_args["path_to_caffemodel"], 
                    caffe.TEST)
    
    
    

# net = caffe.Net("/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/base_prototxt_files/deploy.prototxt",
#                     "/global/homes/r/racah/projects/climate-caffe/2d_semi_sup/notebooks/models/model_iter_2.caffemodel", 
#                     caffe.TEST)

    keys = ['xy_pred',
     'wh_pred','obj_scores_copy','class_scores_copy',
     ]
    bba = BBox_Accuracy(iou_thresh=cl_args["iou_thresh"])
    for ep in range(cl_args["iterations"]):
        blobs =net.forward()

        label = np.float32(net.blobs["label"].data)
        net_out_blobs = [np.float32(net.blobs[k].data) for k in keys]


        net_output = np.concatenate(tuple(net_out_blobs),axis=1)



        bba.update_scores(net_output,label)
	print "Accuracy at iteration %i is %8.4f\n" % (ep,bba.compute_final_accuracy())

    print "Final Accuracy is %8.4f\n" % (bba.compute_final_accuracy())


# In[1]:

#! jupyter nbconvert --to script get_accuracy.ipynb


# In[18]:

#! cp get_accuracy.py /project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/


# In[ ]:



