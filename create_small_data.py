import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
root_dir = './MIO-TCD/'
classes_dir = ["articulated_truck", "background" , "bicycle"  , "bus" , "car" , "motorcycle" , "non-motorized_vehicle" , "pedestrian" , "pickup_truck" , "single_unit_truck" , "work_van"]

dest_dir = './Small/MIO-TCD/'

val_ratio = 0.20


for cls in classes_dir:
    os.makedirs(dest_dir +'/train/' + cls)
    os.makedirs(dest_dir +'/val/' + cls)
    


    # Creating partitions of the data after shuffeling
    src_train = root_dir + 'train'+ cls # Folder to copy images from
    src_val = root_dir + 'val'+ cls
    
    tain_allFileNames = os.listdir(src_train)
    np.random.shuffle(tain_allFileNames)

    val_allFileNames = os.listdir(src_val)
    np.random.shuffle(val_allFileNames)


    tain_allFileNames = np.array(tain_allFileNames)[0:1400]
    val_allFileNames = np.array(val_allFileNames)[0:350]
    

    train_FileNames = [src_train+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src_val+'/' + name for name in val_FileNames.tolist()]

    print('Class = ',cls)
    
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))

    #Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, dest_dir +'/train' + cls)

    for name in val_FileNames:
        shutil.copy(name, dest_dir +'/val' + cls)
