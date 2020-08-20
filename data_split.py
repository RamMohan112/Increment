import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
root_dir = './MIO-TCD/train/'
classes_dir = ["articulated_truck", "background" , "bicycle"  , "bus" , "car" , "motorcycle" , "non-motorized_vehicle" , "pedestrian" , "pickup_truck" , "single_unit_truck" , "work_van"]

dest_dir = './Split/MIO-TCD/'

val_ratio = 0.20


for cls in classes_dir:
    os.makedirs(dest_dir +'/train' + cls)
    os.makedirs(dest_dir +'/val' + cls)
    


    # Creating partitions of the data after shuffeling
    src = root_dir + cls # Folder to copy images from

    allFileNames = os.listdir(src)

    np.random.shuffle(allFileNames)

    train_FileNames, val_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - val_ratio))])


    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]

    print('Class = ',cls)
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))

    # Copy-pasting images
    # for name in train_FileNames:
    #     shutil.copy(name, dest_dir +'/train' + cls)

    # for name in val_FileNames:
    #     shutil.copy(name, dest_dir +'/val' + cls)
