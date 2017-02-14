
# coding: utf-8

# In[1]:

import os
import sys


# In[2]:

def convert_nb_to_script(dir_):
    get_ipython().system(u' jupyter nbconvert --to script $dir_/*.ipynb')

    #.txt means file was open/not saved so converted to .txt not .py
    for fil in os.listdir(dir_):
        assert '.txt' not in fil, "Save all your notebooks first!"
    if dir_ is not ".":
        get_ipython().system(u' mv $dir_/__init__.py $dir_/__init__.pyc')
        get_ipython().system(u' mv $dir_/nbfinder.py $dir_/nbfinder.pyc        ')


    get_ipython().system(u' sed -i.bak \'s/# coding: utf-8/import matplotlib; matplotlib.use("agg")/g\' $dir_/*.py')

    #get rid of IN[] lines
    get_ipython().system(u"sed -i.bak '/# IN\\[*/d' $dir_/*.py")

    #get rid of ipython lines
    get_ipython().system(u"sed -i.bak '/ipython*/d' $dir_/*.py")

    get_ipython().system(u"sed -i.bak '/NotebookFinder*/d' $dir_/*.py")

    get_ipython().system(u" sed -i.bak 's/notebooks./scripts./g' $dir_/*.py")
    #get rid of IN[] lines
    get_ipython().system(u"sed -i.bak '/# In\\[*/d' $dir_/*.py")

    get_ipython().system(u"sed -i.bak '/# coding:*/d' $dir_/*.py")

    get_ipython().system(u"sed -i.bak 's/plt.show()*/pass/g' $dir_/*.py")

    get_ipython().system(u' rm $dir_/*.bak')
    

#     if dir_ is not ".":
#         subdir = "/".join(dir_.split("notebooks/")[1:])
#         if not os.path.exists(os.path.join("./scripts", subdir)):
#             os.makedirs(os.path.join("./scripts", subdir))
#         #move to scripts
#         ! mv $dir_/*.py ./scripts/$subdir
#         ! mv $dir_/__init__.pyc $dir_/__init__.py
#         ! mv $dir_/nbfinder.pyc $dir_/nbfinder.py
#     else:
    get_ipython().system(u' rm $dir_/convert_nb_to_scripts.py')
        


# In[3]:

for dirpath, dirs, files in os.walk("./label_maker_stuff"):
    if not ".ipynb_checkpoint" in dirpath:
        print dirpath
        convert_nb_to_script(dirpath)
convert_nb_to_script(".")


# In[ ]:



