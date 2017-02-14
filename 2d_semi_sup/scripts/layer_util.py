
# coding: utf-8

# In[ ]:

import caffe

from caffe import layers as L, params as P, to_proto

from caffe.proto import caffe_pb2

from caffe.coord_map import crop

import copy

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group, weight_filler={"type":"msra"})
    return L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def deconv_relu(bottom, ks, nout, stride=1, pad=0, group=1, crop=0):
    convolution_param = dict(num_output=nout, kernel_size=ks, stride=stride,pad=pad, group=group,
            bias_term=False)
    deconv = L.Deconvolution(bottom,convolution_param=convolution_param)
    return L.ReLU(deconv, in_place=True)
    
    
    

def get_blob_shape(blob):
    _,fn = write_to_file(blob,filename="tmp")
    net = caffe.Net(fn, caffe.TEST)
    blob_key = net.blobs.keys()[-1]
    blob_obj = net.blobs[blob_key]
    blob_shape = [blob_obj.shape[i] for i in range(len(blob_obj.shape))]
    return blob_shape
    
    
def convert_layer_to_prototxt(netspec):
    return str(netspec.to_proto())

def write_prototxt_str(filename, prototxt_str):
    #print prototxt_str
    with open(filename, 'w') as f:
        f.write(prototxt_str)

def write_to_file(netspec,filename="train"):
    
    prototxt_str = convert_layer_to_prototxt(netspec)

    write_prototxt_str(filename + ".prototxt", prototxt_str)
    
    return prototxt_str, filename + ".prototxt"


#add engine caffe
def add_engine_caffe(pstr, fn):
    splits = pstr.split("operation: SUM")
    newstr=''.join(splits[:-1] + ["operation: SUM\n    engine: CAFFE"] +[splits[-1]])
    write_prototxt_str(fn, newstr)
    return newstr

def get_dummy_shape(data):
    return data.fn.params["shape"]["dim"]

