
# coding: utf-8

# In[1]:

import sys


# In[2]:

import argparse
import os
from os.path import join, basename
import random
import shutil


# In[3]:

cl_args = { "num_nodes": 64,
            "path_to_files": "/global/cscratch1/sd/racah/climate_data/",
            "master_file_path": "/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/master_file_lists/",
            "files_dest_dir": "/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/multinode_runs/file_list_files",
            "prototxt_dest_dir": "/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/multinode_runs/prototxt_files",
            "seed": 3,
           "split_tr_val_test":False,
           "prototxt_split": True,
           "file_split": True,
           "num_crops": 2,
           "solver_filepath":"/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/solver.prototxt",
           "model_filepath":"/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/train.prototxt"}


# In[4]:

def get_cl_args():


    if any(["jupyter" in arg for arg in sys.argv]):
        sys.argv=sys.argv[:1]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in cl_args.iteritems():
        if type(v) is not bool:
            parser.add_argument('--' + k, type=type(v), default=v, help=k)
        else:
            parser.add_argument('--' + k,default=v, action='store_true')

    args = parser.parse_args()
    cl_args.update(args.__dict__)
    return cl_args


# In[5]:

def get_list_of_every_file(basepath):
    images = []
    labels = []
    for dir_ in os.listdir(basepath):
        ims = get_sorted_list_in_dir(join(basepath,dir_, "images"))
        images.extend(ims)
        lbls = get_sorted_list_in_dir(join(basepath,dir_, "labels"))
        labels.extend(lbls)
    file_pairs = zip(images,labels) 
    check_matching(pairs=file_pairs)

    return file_pairs
        


# In[6]:

def check_matching(pairs):
    for im,lbl in pairs:
        im_name = basename(im).split(".nc")[0]
        lbl_name = basename(lbl).split("_label.nc")[0]
        if im_name != lbl_name:
            assert False, "HEY! The pairing is wrong: %s does not match %s"%(im,lbl)


# In[7]:

def shuffle_files(files,seed):
    random.seed(seed)
    random.shuffle(files)
    return files


# In[8]:

def get_sorted_list_in_dir(dir_):
    files = os.listdir(dir_)
    files.sort()
    files = [join(dir_,file_) for file_ in files]
    return files


# In[9]:

def split_train_val_test(file_list, dist={"tr":0.8, "val":0.1, "test": 0.1}):
    num_files = len(file_list)
    ind = 0
    tr_start, val_start, test_start = 0,int(dist["tr"]*num_files), int( (dist["tr"]+ dist["val"]) * num_files)
    tr_files, val_files, test_files = file_list[tr_start:val_start], file_list[val_start: test_start], file_list[test_start:]
    return dict(zip(["tr", "val", "test"],[tr_files, val_files, test_files]))

    
    


# In[10]:

def save_pairs(pair_list,path, file_prefix):
    f_im = open(join(path, file_prefix + "_images_list.txt"), "w")
    f_lbl = open(join(path, file_prefix + "_labels_list.txt"), "w")
    for im, lbl in pair_list:
        f_im.write(im)
        f_lbl.write(lbl)
    f_im.close()
    f_lbl.close()


# In[11]:

def save_pairs_with_crop_indices(pair_list, path, file_prefix, num_crops):
    new_pair_list = []
    crop_list = []
    for pair in pair_list:
        for crop in range(num_crops):
            new_pair_list.append(pair)
            crop_list.append(str(crop))
    save_pairs(pair_list=new_pair_list,path=path, file_prefix=file_prefix)
    save_crops(crop_list, path, file_prefix=file_prefix)
            
    


# In[12]:

def save_crops(crop_list, path, file_prefix):
    with open(join(path, file_prefix + "_crops_list.txt"), "w") as f_crop:
        for crop in crop_list:
            f_crop.write(crop+ "\n")
    
        
    
    


# In[13]:

def get_file_pairs_from_txt_files(im_txt_file, lbl_txt_file):
    ims = open(im_txt_file,"r").readlines()
    lbls = open(lbl_txt_file,"r").readlines()
    file_pairs = zip(ims,lbls)
    check_matching(file_pairs)
    return file_pairs


# In[41]:

def save_n_file_lists(im_txt_file, lbl_txt_file, n, dest_dir, num_crops=0):
    print n
    def make_n_file_pairs_lists(im_txt_file, lbl_txt_file, n):
        file_pairs = get_file_pairs_from_txt_files(im_txt_file=im_txt_file, lbl_txt_file=lbl_txt_file)
        num_files = len(file_pairs)
        num_for_each = num_files / n
        remainder = num_files % n
        tot = 0
        for ind in range(n):
            if ind < remainder:
                
                fp = file_pairs[tot: tot + num_for_each + 1]
                tot += num_for_each + 1
            else:
                fp = file_pairs[tot: tot + num_for_each]
                tot += num_for_each
            yield fp
            
    
    for i, file_pairs in enumerate(make_n_file_pairs_lists(im_txt_file, lbl_txt_file, n)):
        if num_crops > 0:
            save_pairs_with_crop_indices(file_pairs, dest_dir, file_prefix="node_" + str(i) + "_tr" , num_crops=num_crops)
        else:
            save_pairs(file_pairs,dest_dir,file_prefix="node_" + str(i) + "_tr" )
    


# In[27]:

def save_n_model_and_solver_prototxts(model_prototxt_path, solver_prototxt_path, n, file_list_dir, prototxt_dest_dir):
    for i in range(n):
        prefix="node_" + str(i)
        im_file = join(file_list_dir,prefix + "_tr" + "_images_list.txt")
        lbl_file = join(file_list_dir,prefix + "_tr" + "_labels_list.txt")
        crops_file = join(file_list_dir,prefix + "_tr" + "_crops_list.txt") 
        new_model_prototxt_path = make_new_model_prototxt(model_prototxt_path, im_file,lbl_file,crops_file, prototxt_dest_dir, prefix=prefix)
        make_new_solver_prototxt(new_model_prototxt_path, solver_prototxt_path, prototxt_dest_dir, prefix=prefix)
        

    
    


# In[28]:

def make_new_solver_prototxt(new_model_prototxt_path, solver_prototxt_path, prototxt_dest_dir, prefix):
    new_solver_prototxt_path = join(prototxt_dest_dir, prefix + "_" + basename(solver_prototxt_path))
    shutil.copy(solver_prototxt_path, new_solver_prototxt_path)
    replace_solver_with_new_model_prototxt(new_solver_prototxt_path, new_model_prototxt_path)
    


# In[29]:

def make_new_model_prototxt(model_prototxt_path, im_file,lbl_file,crops_file, prototxt_dest_dir, prefix):
    new_model_prototxt_path = join(prototxt_dest_dir, prefix + "_" + basename(model_prototxt_path))
    
    shutil.copy(model_prototxt_path, new_model_prototxt_path)
    replace_im_label_source_in_prototxt(new_model_prototxt_path, im_file, lbl_file, crops_file)
    return new_model_prototxt_path
    
    


# In[36]:

def replace_solver_with_new_model_prototxt(solver_prototxt, model_prototxt):
    with open(solver_prototxt, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "train_net" in line:
                lines[i] = lines[i].split('"')[0] + '"' + model_prototxt + '"' + "\n"
    with open(solver_prototxt, "w") as f:
        f.writelines(lines)


# In[37]:

def replace_im_label_source_in_prototxt(prototxt_path, im_file, lbl_file, crops_file):
    with open(prototxt_path, "r") as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "source: " in line:
                if "crop" not in line:
                    if "image" in line:
                        lines[i] = lines[i].split('"')[0] + '"' + im_file + '"' + "\n"
                    else:
                        lines[i] = lines[i].split('"')[0] + '"' + lbl_file + '"' + "\n"
                else:
                    lines[i] = lines[i].split('"')[0] + '"' + crops_file + '"' + "\n"
    with open(prototxt_path, "w") as f:
        f.writelines(lines)
                


# In[38]:

if __name__ == "__main__":
    cla = get_cl_args()
    #cla["split_tr_val_test"] = True
    if cla["split_tr_val_test"]:
        file_pairs = get_list_of_every_file(cla['path_to_files'])
        file_pairs = shuffle_files(file_pairs,seed=cla["seed"])
        file_pairs_dict = split_train_val_test(file_pairs)
        for typ, pairs in file_pairs_dict.iteritems():
            save_pairs(pairs, cla["master_file_path"],file_prefix=typ)
            
    
    if cla["file_split"]:
        
        mp = cla["master_file_path"]
        im_txt_file, lbl_txt_file = join(mp,"tr_images_list.txt"), join(mp,"tr_labels_list.txt")
        save_n_file_lists(im_txt_file=im_txt_file, 
                          lbl_txt_file=lbl_txt_file, 
                          n=cla["num_nodes"], 
                          dest_dir=cla["files_dest_dir"],
                         num_crops=cla["num_crops"])
    if cla["prototxt_split"]:
        save_n_model_and_solver_prototxts(cla["model_filepath"], cla["solver_filepath"], cla["num_nodes"], cla["files_dest_dir"], cla["prototxt_dest_dir"])
        
    


# In[45]:

#! jupyter nbconvert --to script split_files.ipynb


# In[ ]:



