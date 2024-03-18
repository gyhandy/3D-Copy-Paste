"""
Save .obj for each of the .glb in glb_root to obj_root, based on a json list raw_glbs_list.json
.obj will be in the same folder as .glb
"""
import trimesh
from tqdm import tqdm
from pathlib import Path
import os
import objaverse
import json
import argparse


def main(args):
    with open(args.glb_full_list,
              'r') as fh:
        glb_full_list = json.load(fh)

    raw_root = args.raw_root
    glb_root = args.glb_root
    obj_root = args.obj_root
    for i, file in enumerate(glb_full_list): 
        print('processing', i)
        if i >= int(args.start_i) and i <= int(args.end_i): # in region
            if '.glb' in file: # for all .glb files
                obj_folder = os.path.join(obj_root, file.replace('.glb', ''))
                obj = os.path.join(obj_folder, file.replace('.glb', '.obj'))
                if not os.path.exists(obj): # not convert
                    try:
                        print('processing {}'.format(os.path.join(glb_root, file)))
                        os.makedirs(obj_folder, exist_ok=True)
                        mesh = trimesh.load(os.path.join(glb_root, file))
                        mesh.export(file_type="obj", file_obj=obj, include_texture=True)
                    except:
                        print('############################### Error for object {}'.format(obj))
                        file1 = open(os.path.join(raw_root, 'glb2obj_error_log.txt'), "a")  # append mode
                        file1.write("error {}  \n".format(os.path.join(glb_root, file)))
                        file1.close()



if __name__ == '__main__':
    '''noise data'''
    parser = argparse.ArgumentParser(description='glb2obj')
    parser.add_argument('--glb_full_list', default="/data/objaverse_objects/raw_glbs_list.json", help='save all the downloaded id of .glb')
    parser.add_argument('--raw_root', default="/data/objaverse_objects", help='root path')
    parser.add_argument('--glb_root', default="/data/objaverse_objects/glbs", help='path that save the downloaed .glb')
    parser.add_argument('--obj_root', default="/data/objaverse_objects/objs", help='path that save the converted .obj')
    parser.add_argument('--start_i', help='start index')
    parser.add_argument('--end_i', help='end index')
    args = parser.parse_args()

    main(args)


