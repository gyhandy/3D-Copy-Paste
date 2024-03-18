import numpy as np
from PIL import Image
import os
import scipy
import json
import tqdm


source_root = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval'
z_threshold = 0.04 # point cloud z-axis std threshould to select horizontal plane, 
floor_size_list = {}
# We can classify floor small, medium or large based on their size. In our paper we did not utilize this information during insertion
# But we can insert different size of object based on the floor size if necessary.
s_threshold = 0.7739 # (6500/8954) small size floor area threshould
m_threshold = 2.5477 # (3500/8954) medium size floor area threshould 

for i in range(10335): # for all 3000, 6000 scenes

    img_id = format(i+1, '06d') # SUNRGBD image name, e.g., '000063' '010335'
    print('processing {}'.format(img_id))
    plane_path = os.path.join(source_root, 'plane/{}/plane_statistics.json'.format(img_id))
    if not os.path.exists(plane_path):
        print('can not find plane results of {}'.format(img_id))
        continue


    # load
    with open(plane_path, 'r') as fh:
      plane_statistics_json = json.load(fh) # now the index need to be string


    # find the horizontal plane and floor
    horizontal_plane_info = {}
    floor_index = None
    floor_mean = 10
    for plane_id, plane_sta in plane_statistics_json.items():
        if plane_sta['std'][2] < z_threshold: # small std means horizontal
            horizontal_plane_info[plane_id] = plane_sta
            if plane_sta['mean'][2] < floor_mean: # find the lowest as floor
                floor_mean = plane_sta['mean'][2]
                floor_index = plane_id


    if horizontal_plane_info!={}: # exist horizontal plane
        horizontal_plane_info['floor'] = horizontal_plane_info[floor_index]
        horizontal_plane_info['floor_id'] = floor_index
        x_min = horizontal_plane_info['floor']['min'][0]
        x_max = horizontal_plane_info['floor']['max'][0]
        y_min = horizontal_plane_info['floor']['min'][1]
        y_max = horizontal_plane_info['floor']['max'][1]
        floor_size = (x_max - x_min) * (y_max - y_min)
        horizontal_plane_info['floor']['area'] = floor_size # document the floor size for further filtering
        if floor_size < s_threshold: # small size
            horizontal_plane_info['floor']['size'] = 's'
        elif floor_size > s_threshold and floor_size < m_threshold: # medium size
            horizontal_plane_info['floor']['size'] = 'm'
        else: # large size
            horizontal_plane_info['floor']['size'] = 'l'
        SATISFY = True
        # check if the floor is a real floor instead of ceiling
        for plane_id, plane_sta in plane_statistics_json.items():
            if plane_sta['mean'][2] < horizontal_plane_info['floor']['mean'][2]:  # if there exist another plane has lower mean than floor
                with open(os.path.join(source_root, 'plane/no_floor_scene_threshold{}.json'.format(z_threshold)),
                          "a") as outfile:
                    json.dump(img_id + '\n', outfile, indent=4)
                print('############################### no_floor for image {}'.format(img_id))
                SATISFY = False # not a good floor
                break
        if SATISFY: # save only when floor pass the filtering
            floor_size_list[img_id] = floor_size
            # save
            root_path = os.path.join(source_root, 'plane/{}'.format(img_id))
            # with open(os.path.join(root_path, 'floor_plane_statistics.json'), "w") as outfile:
            #     json.dump(horizontal_plane_info, outfile, indent=4)
            with open(os.path.join(root_path, 'floor_plane_statistics_noceiling_threshold{}.json'.format(z_threshold)), "w") as outfile:
                json.dump(horizontal_plane_info, outfile, indent=4)
    else: # no floor
        with open(os.path.join(source_root, 'plane/no_floor_scene_threshold{}.json'.format(z_threshold)), "a") as outfile:
            json.dump(img_id + '\n', outfile, indent=4)
        print('############################### no_floor for image {}'.format(img_id))
sorted_floor_size_list = sorted(floor_size_list.items(), key=lambda x: x[1], reverse=True)
with open(os.path.join(source_root, 'plane/sorted_floor_size_threshold{}.json'.format(z_threshold)), "w") as outfile:
    json.dump(sorted_floor_size_list, outfile, indent=4)







