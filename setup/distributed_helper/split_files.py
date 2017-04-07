
# coding: utf-8

# In[1]:

import sys


# In[2]:

import argparse
import os
from os.path import join, basename,exists
import random
import shutil


# In[3]:

def get_cl_args(cl_args):


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


# In[4]:

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
        


# In[5]:

def check_matching(pairs):
    for im,lbl in pairs:
        im_name = basename(im).split(".nc")[0]
        lbl_name = basename(lbl).split("_label.nc")[0]
        if im_name != lbl_name:
            assert False, "HEY! The pairing is wrong: %s does not match %s"%(im,lbl)


# In[6]:

def shuffle_files(files,seed):
    random.seed(seed)
    random.shuffle(files)
    return files


# In[7]:

def get_sorted_list_in_dir(dir_):
    files = os.listdir(dir_)
    files.sort()
    files = [join(dir_,file_) for file_ in files]
    return files


# In[8]:

def split_train_val_test(file_list, dist={"tr":0.8, "val":0.1, "test": 0.1}):
    num_files = len(file_list)
    ind = 0
    tr_start, val_start, test_start = 0,int(dist["tr"]*num_files), int( (dist["tr"]+ dist["val"]) * num_files)
    tr_files, val_files, test_files = file_list[tr_start:val_start], file_list[val_start: test_start], file_list[test_start:]
    return dict(zip(["tr", "val", "test"],[tr_files, val_files, test_files]))

    
    


# In[9]:

def save_pairs(pair_list,path, file_prefix):
    if not exists(path):
        os.mkdir(path)
    f_im = open(join(path, file_prefix + "_images_list.txt"), "w")
    f_lbl = open(join(path, file_prefix + "_labels_list.txt"), "w")
    for im, lbl in pair_list:
        im = im.replace("\n","")
        lbl = lbl.replace("\n", "")
        f_im.write(im + "\n" )
        f_lbl.write(lbl + "\n")
    f_im.close()
    f_lbl.close()


# In[10]:

def save_pairs_with_crop_indices(pair_list, path, file_prefix, num_crops):
    new_pair_list = []
    crop_list = []
    for pair in pair_list:
        for crop in range(num_crops):
            new_pair_list.append(pair)
            crop_list.append(str(crop))
    save_pairs(pair_list=new_pair_list,path=path, file_prefix=file_prefix)
    save_crops(crop_list, path, file_prefix=file_prefix)
            
    


# In[11]:

def save_crops(crop_list, path, file_prefix):
    if not exists(path):
        os.mkdir(path)
    with open(join(path, file_prefix + "_crops_list.txt"), "w") as f_crop:
        for crop in crop_list:
            f_crop.write(crop+ "\n")
    
        
    
    


# In[12]:

def get_file_pairs_from_txt_files(im_txt_file, lbl_txt_file):
    ims = open(im_txt_file,"r").readlines()
    lbls = open(lbl_txt_file,"r").readlines()
    file_pairs = zip(ims,lbls)
    check_matching(file_pairs)
    return file_pairs


# In[13]:

def make_filelist_dir(dest_dir,n, master_path):
#     dataset_name = basename(master_path)
#     dir_ = join(dest_dir,dataset_name,str(n),"file_list_files")
    return dest_dir


# In[14]:

def make_prototxt_dir(dest_dir,n, master_path):
#     dataset_name = basename(master_path)
#     dest_dir = join(dest_dir,dataset_name,str(n),"prototxt_files")
    return dest_dir


# In[15]:

def save_n_file_lists(im_txt_file, lbl_txt_file, n, dest_dir, num_crops, master_file_path, typ="tr", constant_num_for_each=None):
    dest_dir = make_filelist_dir(dest_dir, n, master_file_path)
    if not exists(dest_dir):
        os.makedirs(dest_dir)
    print n
    def make_n_file_pairs_lists(im_txt_file, lbl_txt_file, n):
        file_pairs = get_file_pairs_from_txt_files(im_txt_file=im_txt_file, lbl_txt_file=lbl_txt_file)
        num_files = len(file_pairs)
        if constant_num_for_each:
            num_for_each = constant_num_for_each
        else:
            num_for_each = num_files / n
        remainder = num_files % n
        tot = 0
        for ind in range(n):
            if ind < remainder and constant_num_for_each is None:
                
                fp = file_pairs[tot: tot + num_for_each + 1]
                tot += num_for_each + 1
            else:
                fp = file_pairs[tot: tot + num_for_each]
                tot += num_for_each
            yield fp
            
    
    for i, file_pairs in enumerate(make_n_file_pairs_lists(im_txt_file, lbl_txt_file, n)):
        if num_crops > 0:
            save_pairs_with_crop_indices(file_pairs, dest_dir, file_prefix="node_" + str(i+1) + "_" + typ , num_crops=num_crops)
        else:
            save_pairs(file_pairs,dest_dir,file_prefix="node_" + str(i+1) + "_" + typ)
    


# In[16]:

def save_n_model_and_solver_prototxts(model_prototxt_path, solver_prototxt_path, n, file_list_dir, prototxt_dest_dir, master_path):
    prototxt_dest_dir = make_prototxt_dir(prototxt_dest_dir,n, master_path)
    if not exists(prototxt_dest_dir):
        os.makedirs(prototxt_dest_dir)
    for i in range(n):
    
        prefix="node_" + str(i+1)
        im_file = join(file_list_dir,prefix + "_tr" + "_images_list.txt")
        lbl_file = join(file_list_dir,prefix + "_tr" + "_labels_list.txt")
        crops_file = join(file_list_dir,prefix + "_tr" + "_crops_list.txt")
        tr_files = [im_file,lbl_file,crops_file]
        
        val_im_file = join(file_list_dir,prefix + "_val" + "_images_list.txt")
        val_lbl_file = join(file_list_dir,prefix + "_val" + "_labels_list.txt")
        val_crops_file = join(file_list_dir,prefix + "_val" + "_crops_list.txt") 
        val_files = [val_im_file,val_lbl_file,val_crops_file]
        new_model_prototxt_path = make_new_model_prototxt(model_prototxt_path,tr_files, val_files, prototxt_dest_dir, prefix=prefix)
        make_new_solver_prototxt(new_model_prototxt_path, solver_prototxt_path, prototxt_dest_dir, prefix=prefix)
        

    
    


# In[17]:

def make_new_solver_prototxt(new_model_prototxt_path, solver_prototxt_path, prototxt_dest_dir, prefix):
    new_solver_prototxt_path = join(prototxt_dest_dir, prefix + "_" + basename(solver_prototxt_path))
    shutil.copy(solver_prototxt_path, new_solver_prototxt_path)
    replace_solver_with_new_model_prototxt(new_solver_prototxt_path, new_model_prototxt_path)
    


# In[18]:

def make_new_model_prototxt(model_prototxt_path,tr_files, val_files, prototxt_dest_dir, prefix):
    new_model_prototxt_path = join(prototxt_dest_dir, prefix + "_" + basename(model_prototxt_path))
    
    shutil.copy(model_prototxt_path, new_model_prototxt_path)
    replace_im_label_source_in_prototxt(new_model_prototxt_path, tr_files, val_files)
    return new_model_prototxt_path
    
    


# In[19]:

def replace_solver_with_new_model_prototxt(solver_prototxt, model_prototxt):
    with open(solver_prototxt, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "train_net" in line or "test_net" in line:
                lines[i] = lines[i].split('"')[0] + '"' + model_prototxt + '"' + "\n"
    with open(solver_prototxt, "w") as f:
        f.writelines(lines)


# In[20]:

def replace_im_label_source_in_prototxt(prototxt_path, tr_files, val_files):
    tr_im_file,tr_lbl_file,tr_crops_file = tr_files
    val_im_file,val_lbl_file,val_crops_file = val_files
    with open(prototxt_path, "r") as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "source: " in line:
                if "crop" not in line:
                    if "image" in line:
                        if "TRAIN" in lines[i-3]:
                            lines[i] = lines[i].split('"')[0] + '"' + tr_im_file + '"' + "\n"
                        else:
                            lines[i] = lines[i].split('"')[0] + '"' + val_im_file + '"' + "\n"
                    else:
                        if "TRAIN" in lines[i-3]:
                            lines[i] = lines[i].split('"')[0] + '"' + tr_lbl_file + '"' + "\n"
                        else:
                            lines[i] = lines[i].split('"')[0] + '"' + val_lbl_file + '"' + "\n"
                else:
                    
                        if "TEST" in lines[i-14] or "TEST" in lines[i-24]:
                            lines[i] = lines[i].split('"')[0] + '"' + val_crops_file + '"' + "\n"
                        else:
                            lines[i] = lines[i].split('"')[0] + '"' + tr_crops_file + '"' + "\n"
    with open(prototxt_path, "w") as f:
        f.writelines(lines)
                


# In[24]:

if __name__ == "__main__":
    basedir="/project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup/"
    cl_args = { "num_nodes": 32,
                "path_to_files": join(basedir, "/global/cscratch1/sd/racah/climate_data/"),
                "master_file_path": join(basedir, "master_file_lists/small_medium/"),
                "files_dest_dir": join(basedir,"multinode_runs/weak_test"),
                "prototxt_dest_dir": join(basedir,"multinode_runs/"),
                "seed": 3,
                "split_tr_val_test":False,
                "prototxt_split": True,
                "file_split": True,
                "num_crops": 1,
                "solver_filepath":join(basedir, "base_prototxt_files/solver_vanilla.prototxt"),
                "model_filepath":join(basedir"base_prototxt_files/train_vanilla.prototxt", 
                "num_total_files": -1,
                "constant_num_for_each": 4
              }
    
    
    
    cla = get_cl_args(cl_args)
    if cla["split_tr_val_test"]:
        file_pairs = get_list_of_every_file(cla['path_to_files'])
        file_pairs = shuffle_files(file_pairs,seed=cla["seed"])
        if cl_args["num_total_files"] != -1:
            file_pairs = file_pairs[:cl_args["num_total_files"]]
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
                         num_crops=cla["num_crops"],master_file_path=cla["master_file_path"], typ="tr", constant_num_for_each=cla["constant_num_for_each"])
        
        im_txt_file, lbl_txt_file = join(mp,"val_images_list.txt"), join(mp,"val_labels_list.txt")
        save_n_file_lists(im_txt_file=im_txt_file, 
                          lbl_txt_file=lbl_txt_file, 
                          n=cla["num_nodes"], 
                          dest_dir=cla["files_dest_dir"],
                         num_crops=cla["num_crops"],master_file_path=cla["master_file_path"],typ="val", constant_num_for_each=cla["constant_num_for_each"])
        
        
    if cla["prototxt_split"]:
        save_n_model_and_solver_prototxts(cla["model_filepath"], 
                                          cla["solver_filepath"], 
                                          cla["num_nodes"], 
                                          make_filelist_dir(cla["files_dest_dir"], cla["num_nodes"], cla["master_file_path"]),
                                          cla["prototxt_dest_dir"],cla["master_file_path"])
        
    


# In[19]:

get_ipython().system(u' jupyter nbconvert --to script split_files.ipynb')


# In[20]:

get_ipython().system(u' mv split_files.py /project/projectdirs/dasrepo/gordon_bell/deep_learning/networks/climate/2d_semi_sup')


# In[ ]:



