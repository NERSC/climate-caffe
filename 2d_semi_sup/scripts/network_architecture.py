
# coding: utf-8

# In[ ]:

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
    
def encode(n, conv, nfilters_list, num_layers):
    encoder_blobs = [conv]
    for i in range(num_layers):
        nfilters = nfilters_list[i]
        conv = conv_relu(conv, ks=5, nout=nfilters, pad=2, stride=2)
        encoder_blobs.append(conv)
    n.encoder = conv
    return n, encoder_blobs
    
def decode(n, encoder, encoder_blobs, nfilters_list, num_layers, num_input_channels):
    #remove last layer and correpsonding number of filters b/c we don't use the last number in reverse
    nfilters_list.pop()
    encoder_blobs.pop()
    
    #reverse the list b/c decoding goes in reverse
    nfilters_list.reverse()
    encoder_blobs.reverse()
    
    # add the channel size of input data for full reconstruction
    nfilters_list.append(num_input_channels)

    conv = encoder
    for i in range(num_layers):
        nfilters = nfilters_list[i]
        conv = deconv_relu(conv,5, nfilters, stride=2)
        conv = L.Crop(conv, encoder_blobs[i], axis=2,offset=1)
    n.decoder = conv
    return n

def bbox_reg(n):
    n.gxy, n.gwh, n.gobj, n.gcls = L.Slice(n.label, slice_point=[2,4,6], ntop=4)

    
    n.cls = L.ArgMax(n.gls, axis=1)
    
    n.class_scores = conv_relu(n.encoder,ks=3,pad=1,nout=num_classes)
    n.L_cls = L.SoftmaxWithLoss(n.class_scores, n.cls, ignore_label=0)

    n.conv2 = L.Convolution(n.encoder,kernel_size=3,pad=1, stride=1, num_output=lbl_ch[1])

    n.conv2 = L.Sigmoid(n.conv2)
    n.conv3 = L.Convolution(n.encoder,kernel_size=3,pad=1,stride=1, num_output=lbl_ch[2])

    n.conv3 = L.Sigmoid(n.conv3)
    n.yolo = L.Concat(n.conv1,n.conv2, n.conv3)
    return n
    
    
def create_net(n, data,nfilters_list, num_classes, num_input_channels):

    num_layers = len(nfilters_list)
    conv = data
    n, encoder_blobs = encode(n, conv, nfilters_list, num_layers)
    
    n = decode(n, n.encoder, encoder_blobs, nfilters_list, num_layers, num_input_channels)
   

    rec = n.decoder
    
    n=bbox_reg(n, num_classes)

    return n

