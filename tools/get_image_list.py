import os
import sys
import glob

from natsort import natsorted

def get_image_lists(folders):
    image_lists=[]
    for folder in folders:
        image_lists.append(natsorted(glob.glob(os.path.join(folder,"*.jpg"))))
    return list(zip(*image_lists))