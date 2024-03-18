import os
import numpy as np
import scipy
import random
import json


root_path = '/Users/yunhaoge/PycharmProjects/real2sim2real/data/'
scene_pointcloud_files = 'SUNRGBD_data/pointcloud'
output_path = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/SUNRGB_object_statistics.json'

interested = ["bed", "table", "sofa", "chair", "desk", "dresser", "night_stand", "nightstand", "bookshelf", "bookcase", "book_shelf", "bookcase",
              "bathtub", "toilet"]

interested_objects = {
    k: []
    for k in interested
}

# training data information
with open(os.path.join(root_path, 'SUNRGBD_data/train_data_idx.txt')) as f:
    train_data_ids_raw = f.readlines()

train_data_ids = ["{:06d}".format(int(id.split('\n')[0])) for id in train_data_ids_raw]  # change format


for scene_id in train_data_ids: # do the first 10 for debug, for each scene
    print('Finish image id ', scene_id)


    with open(os.path.join(root_path, 'SUNRGBD_data/label', '{}.txt'.format(scene_id))) as f:
        lines = f.readlines()

    ori_GT_object_info_dict = {}
    ori_GT_object_info_list = []

    if len(lines) == 0:  # do not exist objects, load a templete
        with open(os.path.join(root_path, 'SUNRGBD_data/label', '000063.txt'.format(scene_id))) as f:
            lines = f.readlines()

    for line in lines:  # for each GT object
        raw_info_list = [line.split('\n')[0].split(' ')][0]
        class_name = raw_info_list[0]
        info_list = [float(ele) for ele in raw_info_list[1:]]
        if class_name not in ori_GT_object_info_dict.keys():  # each object insert only one
            ori_GT_object_info_dict[class_name] = info_list
            ori_GT_object_info_list.append(info_list)  # easy to compute min max
    for obj_class, obj_info in ori_GT_object_info_dict.items():
        if obj_class in interested:
            interested_objects[obj_class].append(obj_info[9]) # only about height, the size

# post process, 
# merge 'night_stand' into 'nightstand'
nightstand_list = interested_objects['nightstand']
night_stand_list = interested_objects['night_stand']
interested_objects['nightstand'] = nightstand_list + night_stand_list
del interested_objects['night_stand']

# merge 'bookcase' into 'bookshelf'
bookshelf_list = interested_objects['bookshelf']
bookcase_list = interested_objects['bookcase']
interested_objects['bookshelf'] = bookshelf_list + bookcase_list
del interested_objects['bookcase']
del interested_objects['book_shelf']

# save
interested_objects_statistics = {}
for key, value in interested_objects.items():
    interested_objects_statistics[key] = []
    interested_objects_statistics[key].append(np.mean(value))
    interested_objects_statistics[key].append(np.std(value))

with open(output_path, 'w') as outfile:
    json.dump(interested_objects_statistics, outfile, indent=4)
