
# coding: utf-8

# In[3]:

import numpy as np
from numpy import *


# In[19]:

from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


# In[20]:

from matplotlib import patches


# In[ ]:




# In[6]:

def non_maximal_suppression(box_list, conf,overlap):
    score=conf
    XMIN, XMAX, YMIN,YMAX = range(4)
    x2 = box_list[:, XMAX]
    x1 = box_list[:, XMIN]
    y2 = box_list[:, YMAX]
    y1 = box_list[:, YMIN]
    
    area = (x2-x1+1)*(y2-y1+1)

    #vals = sort(score)
    I = argsort(score)
    pick = []
    count = 1
    while (I.size!=0):
        #print "Iteration:",count
        last = I.size
        i = I[last-1]
        pick.append(i)
        suppress = [last-1]
        for pos in range(last-1):
            j = I[pos]
            xx1 = max(x1[i],x1[j])
            yy1 = max(y1[i],y1[j])
            xx2 = min(x2[i],x2[j])
            yy2 = min(y2[i],y2[j])
            w = xx2-xx1+1
            h = yy2-yy1+1
            if (w>0 and h>0):
                o = w*h/area[j]
                
                if (o >overlap):
                    print "Overlap is",o
                    suppress.append(pos)
        I = delete(I,suppress)
        count = count + 1
    return box_list[pick]
    


# In[7]:




def convert_xy_min_max_tensor_to_box_list(xy_min_max):
    #expects a 4xXxY tensor

    xdim,ydim = xy_min_max.shape[1:]

    xy_min_max = np.transpose(xy_min_max, axes=(1,2,0))
    box_list = xy_min_max.reshape(xdim*ydim,4)
    return box_list

def calc_score(cls, xy_min_max_pred, xy_min_max_gt, conf_pred, conf_gt, cls_pred, cls_gt, iou_thresh=0.1 ,im=None):
#     for g in [xy_min_max_pred, xy_min_max_gt, conf_pred, conf_gt, cls_pred, cls_gt]:
#         print g.shape
    # if there is no object then we don't penalize cuz its unlabelled example
    if np.all(conf_gt == 0.0):
        return [],[]
    
    box_list_pred = convert_xy_min_max_tensor_to_box_list(xy_min_max_pred)
    
    box_list_gt = convert_xy_min_max_tensor_to_box_list(xy_min_max_gt)
    
    #flatten
    flat_cls_gt = cls_gt.reshape(cls_gt.shape[0] * cls_gt.shape[1] )
    flat_conf_gt = conf_gt.reshape(conf_gt.shape[0] * conf_gt.shape[1])
    flat_cls_pred = cls_pred.reshape(cls_pred.shape[0] * cls_pred.shape[1])
    flat_conf_pred = conf_pred.reshape(conf_pred.shape[0] * conf_pred.shape[1])
    
    #get only the boxes where there is an object
    box_list_gt = box_list_gt[flat_conf_gt > 0.]
    
    
    #should be covered by above
    if box_list_gt.shape[0] == 0:
        return [],[]
    flat_cls_gt = flat_cls_gt[ flat_conf_gt > 0]
    box_list_gt = box_list_gt[flat_cls_gt == cls]

    if box_list_gt.shape[0] == 0:
        return [],[]

    #sort by confidence in descending order
    conf_ind = np.argsort(-flat_conf_pred)
    box_list_pred = box_list_pred[conf_ind]
    conf_list = flat_conf_pred[conf_ind]
    flat_cls_pred = flat_cls_pred[conf_ind]
    
    #get only nonzero confidences
    
    box_list_pred = box_list_pred[conf_list > 0.0]
    flat_cls_pred = flat_cls_pred[conf_list > 0.0]
    conf_list = conf_list[conf_list > 0.0]
    




    #filter so just cls remains
    box_list_pred = box_list_pred[flat_cls_pred == cls]
    conf_list = conf_list[flat_cls_pred == cls]
    box_list_pred = non_maximal_suppression(box_list_pred,conf=conf_list,overlap=0.5)
    
    #get top 10 boxes
    box_list_pred = box_list_pred[:10] if box_list_pred.shape[0] >=10 else box_list_pred
    conf_list = conf_list[:10] if conf_list.shape[0] >=10 else conf_list
    
    print cls,conf_list
    
    
    num_gt_boxes = box_list_gt.shape[0]
    
    #boolean array for whether gt box has been matched yet
    gt_box_picked_yet = np.zeros(num_gt_boxes)
    
    
    y_true = []
    y_score = []
    
    if im is not None:
        plot_boxes(im,box_list_pred, box_list_gt, cls)
    
    for box_ind, box_pred in enumerate(box_list_pred):
        iou, iou_ind = get_iou_box_list_of_boxes(box_pred,box_list_gt)
        #add confidence to y_score
        y_score.append(conf_list[box_ind])
        print "box_pred: ", box_pred, "box_gt: ", box_list_gt, "iou: ", iou
        if iou > iou_thresh:
            

            # gt_box can only match with one thang
            if not gt_box_picked_yet[iou_ind]:
                # true positive
                y_true.append(1.)
                gt_box_picked_yet[iou_ind] = 1.

            # gt box already picked
            else:
                # false positive
                y_true.append(0.)

        # box does not overlap more than 0.5 with gt box
        else:

            y_true.append(0.)

    return y_true, y_score


    


# In[8]:

# bba = BBox_Accuracy()
# for i in range(20):
#     net_output = np.concatenate((6*np.random.rand(4,4,24,24)-3, np.random.rand(4,2,24,24), np.random.rand(4,4,24,24)),axis=1)
#     label = np.concatenate((6*np.random.rand(4,4,24,24)-3, np.random.randint(2,size=(4,1,24,24)), np.random.randint(1,5,size=(4,1,24,24))),axis=1)
#     bba.update_scores(net_output, label)
# print bba.compute_final_accuracy()


# In[9]:

def get_obj_cls_scores_from_net_output(output):

    obj = output[:,4]
    cls = np.argmax(output[:,5:], axis=1)
    #print cls.shape
    #obj,cls = np.expand_dims(obj, axis=1), np.expand_dims(cls, axis=1)
    return obj, cls


# In[10]:

def get_obj_cls_scores_from_label(label):
    cls = label[:,-1] - 1
    obj = label[:,-2]
    #obj, cls = np.expand_dims(obj, axis=1), np.expand_dims(cls, axis=1)
    return obj, cls


# In[11]:

def get_iou_box_list_of_boxes(box1,box_list):
        XMIN, XMAX, YMIN,YMAX = range(4)
        xmaxes = box_list[:, XMAX]
        xmins = box_list[:, XMIN]
        ymaxes = box_list[:, YMAX]
        ymins = box_list[:, YMIN]
        
        xmin1,xmax1,ymin1, ymax1 = box1
        
        
        ixmin = np.maximum(xmins, xmin1)
        iymin = np.maximum(ymins, ymin1)
        ixmax = np.minimum(xmaxes, xmax1)
        iymax = np.minimum(ymaxes, ymax1)
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((xmax1 -xmin1 + 1.) * (ymax1 - ymin1 + 1.) +
               (xmaxes - xmins + 1.) *
               (ymaxes - ymins + 1.) - inters)

        ious = inters / uni
        max_iou = np.max(ious)
        ind_max_iou = np.argmax(ious)
        return max_iou, ind_max_iou


# In[12]:

def unparametrize(xy,wh, scale_factor=32):
    xy_raw = unparametrize_xy(xy,scale_factor)
    wh_raw = unparametrize_wh(wh,scale_factor)
    return xy_raw, wh_raw
    

def unparametrize_xy(xy,scale_factor):
    #xy should be num_ex,2,24,24
    num_ex = xy.shape[0]
    xdim = xy.shape[-2]
    ydim = xy.shape[-1]
    xarange = np.arange(xdim)
    yarange = np.arange(ydim)
    xgrid, ygrid = np.meshgrid(xarange, yarange)
    grid = np.vstack((xgrid, ygrid)).reshape(2,xdim,ydim)
    inds = np.zeros_like(xy) # np.repeat(grid,[num_ex,num_ex],0)#.reshape(num_ex,2,xdim,ydim)
    inds[:,0] = xgrid
    inds[:,1] = ygrid
    
    #xy is the normalized offsets, so we add by the ind to get unoffset, then multiply by scale factor to get raw
    raw_normalized = xy + inds
    xy_raw = scale_factor * raw_normalized
    return xy_raw
    
    

def unparametrize_wh(wh,scale_factor):
    wh_unlog = np.power(2*np.ones_like(wh), wh)
    wh_raw = scale_factor * wh_unlog
    return wh_raw
    

def convert_parametrization_to_xy_min_max(xy,wh):
    xy_raw, wh_raw = unparametrize(xy,wh, scale_factor=32)
    xmin_max_y_min_max = convert_from_xy_wh_to_min_max(xy_raw,wh_raw)
    return xmin_max_y_min_max
    
    
def convert_from_xy_wh_to_min_max(xy_raw,wh_raw):
        
        x = xy_raw[:,0]
        y = xy_raw[:,1]
        w = wh_raw[:,0]
        h = wh_raw[:,1]
        xmin,xmax = np.maximum(np.zeros_like(x),x - w / 2.), x + w/ 2.
        ymin, ymax = np.maximum(np.zeros_like(y),y - h / 2.), y+ h/ 2.
        xmin, xmax, ymin, ymax = [np.expand_dims(arr,axis=1) for arr in [xmin, xmax, ymin, ymax]]
        return np.concatenate((xmin, xmax, ymin, ymax),axis=1)
    
    


# In[13]:

def get_iou_box_pair(box1,box2):
    xmin1, xmax1, ymin1, ymax1 = box1
    xmin2, xmax2, ymin2, ymax2 = box2
    
     #min of the xmaxes
    min_xmax = min(xmax1, xmax2 )

    #max of the xmins
    max_xmin = max(xmin1, xmin2 )

    #min of the ymaxes
    min_ymax = min(ymax1, ymax2 )

    #max of the ymins
    max_ymin = max(ymin1, ymin2 )

    xdiff = min_xmax - max_xmin

    ydiff = min_ymax - max_ymin

    inters = max(0, xdiff) * max(0, ydiff)

    def get_area(x_min, x_max, y_min, y_max):
        area = (x_max - x_min + 1) * (y_max - y_min + 1)
        assert area > 0.0, "area should be greater than zero!"
        return area

    area1 = get_area(xmin1, xmax1, ymin1, ymax1)
    area2 = get_area(xmin2, xmax2, ymin2, ymax2)

    union = area1 + area2 - inters
    iou = inters / float(union)
    return iou
    
    
    
    


# In[14]:

def get_xy_wh(output):
    xy = output[:,:2]
    wh = output[:,2:4]
    return xy,wh
    


# In[15]:

def test_unparametrize():
    xy = np.random.rand(4,2,24,24)

    xy_raw = unparametrize_xy(xy, 32)

    abs_range = np.abs(np.log2(1./32)) + np.log2(24)
    small = np.log2(1./32)

    wh = abs_range* np.random.rand(4,2,24,24) + small

    wh_raw = unparametrize_wh(wh,32)
    assert np.min(wh_raw) > 0.
    assert np.max(wh_raw) < 768.
    assert np.min(xy_raw) > 0.
    assert np.max(xy_raw) < 768.
    return xy_raw, wh_raw


# In[16]:

xy_raw,wh_raw = test_unparametrize()


# In[17]:

def test_convert():
    xy_raw,wh_raw = test_unparametrize()
    assert np.all(xy_raw >= 0.0)
    assert np.all(wh_raw >= 0.0)
    return convert_from_xy_wh_to_min_max(xy_raw,wh_raw)
    


# In[18]:

class BBox_Accuracy(object):
    def __init__(self, num_classes=4, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.num_classes = num_classes
        self.y_score ={cls:[] for cls in range(num_classes)}
        self.y_true = {cls:[] for cls in range(num_classes)}
        
    
    def update_scores(self,net_output, label, im=None):
        self._compute_score(net_output, label,im)

    def _compute_score(self,net_output, label,im):
        xy_pred, wh_pred = get_xy_wh(net_output)
        conf_pred, cls_pred = get_obj_cls_scores_from_net_output(net_output)
        xy_gt, wh_gt = get_xy_wh(label)
        conf_gt, cls_gt = get_obj_cls_scores_from_label(label)
        xy_min_max_gt = convert_parametrization_to_xy_min_max(xy_gt,wh_gt)
        xy_min_max_pred = convert_parametrization_to_xy_min_max(xy_pred,wh_pred)


        for ex_ind in range(net_output.shape[0]):
            for cls in range(self.num_classes):
                y_true, y_score = calc_score(cls, xy_min_max_pred[ex_ind], xy_min_max_gt[ex_ind], conf_pred[ex_ind], conf_gt[ex_ind], cls_pred[ex_ind], cls_gt[ex_ind], iou_thresh=self.iou_thresh, im=im[ex_ind] )
                self.y_score[cls].extend(y_score)
                self.y_true[cls].extend(y_true)


    def compute_final_accuracy(self):
        APs = self.compute_final_APs()
        mAP = self.compute_mAP(APs)
        return mAP


    def compute_mAP(self,APs):
        filtered_aps = [ap for ap in APs.values() if ap>=0.0]
        if len(filtered_aps) > 0:
            return np.mean(filtered_aps)
        else:
            return 0.0

    def compute_final_APs(self):
        APs = {}
        for cls in self.y_score.keys():
            if len(self.y_score[cls]) > 0:
                if np.any(self.y_score[cls]):
                    APs[cls] = average_precision_score(self.y_true[cls], self.y_score[cls])
                else:
                    APs[cls] = -1
            else:
                APs[cls] = -1
        return APs

    


# In[17]:

def plot_boxes(im,box_list_pred, box_list_gt, cls):
    sp = plt.subplot(111)
    sp.imshow(im)
    for box in box_list_pred:
        add_bbox(sp,box,color="r")
    for box in box_list_gt:
        add_bbox(sp, box, color="g")
    plt.show()
    


# In[18]:

def add_bbox(subplot, bbox, color):
        #box of form center x,y  w,h
        x1,x2,y1,y2, conf1, cls = bbox
        
        h = y2 - y1 +1
        w = x2 - x1 +1
        xcent, ycent = ( y2 - h / 2., x2 - w / 2.)
        subplot.add_patch(patches.Rectangle(
        xy=( xcent,ycent),
        width=h,
        height=w, lw=2,
        fill=False, color=color))
        #subplot.text(ycent,xcent,classes[cls], fontdict={"color":color})


# In[24]:

#! jupyter nbconvert --to script accuracy.ipynb


# In[20]:

#! cp accuracy.py /project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup


# In[ ]:



