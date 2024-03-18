# Conduct 3D Copy Paste with Blender
# First render inserted object images with groundtruth 3D bounding box, then paste it on the original scene images, generate the final images.

import bpy
from mathutils import *
from math import *
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import json
import os
from shapely import Polygon
import cv2
import argparse

def load_insert_object(filepath):
    '''
    Foreground loading
    '''
    # Load glb foreground, here we load a desk
    # bpy.ops.import_scene.gltf(filepath=filepath) # good
    bpy.ops.import_scene.obj(filepath=filepath)  # good

    # Select all the imported objects
    objects_to_combine = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            objects_to_combine.append(obj)

    # Combine the selected objects into one object
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects_to_combine:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects_to_combine[0]
    bpy.ops.object.join()

    # Rename the combined object
    inobject = bpy.context.object
    inobject.name = "inobject"

    # Set the origin position of the combined object
    bpy.ops.object.select_all(action='DESELECT')
    inobject.select_set(True)
    bpy.context.view_layer.objects.active = inobject
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # build bounding box
    bpy.ops.mesh.boundbox_add()

    bbox_name = None
    for ele in bpy.data.objects.keys():
        if 'BoundingBox' in ele:
            bbox_name = ele
            break

    bbox = bpy.data.objects[bbox_name]

    return inobject, bbox


def select_inserted_object(class_name):
    '''
    Randomly select one object given one objaverse class_name, for selected large set
    '''
    candidates_dict = objaverse_objects_dict[class_name]
    uid = random.choice(candidates_dict)
    obj_path = os.path.join(obj_root_path, uid, uid+'.obj')
    return obj_path


def select_inserted_object_class(objaverse_object_name_list_x, ori_cate_list, insertion_mode):
    '''
    Randomly select one class given class list and insertion_mode
    insertion_mode: random or context
    '''
    if insertion_mode == 'random':
        samples = random.choice(objaverse_object_name_list_x)
        return samples
    elif insertion_mode == 'context':
        candidate_class_list = []
        for cate in ori_cate_list:
            if cate in objaverse_object_name_list_x:
                candidate_class_list.append(cate)
        if candidate_class_list == []: # empty
            candidate_class_list = objaverse_object_name_list_x
        samples = random.choice(candidate_class_list)
        return samples

def iou_rotated_rectangles(corners1, corners2):
    """
    Compute the intersection over union (IOU) of two rotated rectangles, represented by their four corners.

    Args:
        corners1: A list of four tuples, where each tuple contains the x and y coordinates of a corner of the first rectangle.
        corners2: A list of four tuples, where each tuple contains the x and y coordinates of a corner of the second rectangle.

    Returns:
        The IOU of the two rectangles.
    """
    # Calculate the areas of the rectangles
    area1 = polygon_area(corners1)
    area2 = polygon_area(corners2)

    # Calculate the intersection of the rectangles
    intersection_corners = polygon_intersection(corners1, corners2)
    intersection_area = polygon_area(intersection_corners)

    # Calculate the IOU
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou


def polygon_area(corners):
    """
    Calculate the signed area of a polygon using the shoelace formula.

    Args:
        corners: A list of tuples, where each tuple contains the x and y coordinates of a corner of the polygon.

    Returns:
        The signed area of the polygon.
    """
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_intersection(corners1, corners2):
    """
    Calculate the intersection of two polygons.

    Args:
        corners1: A list of tuples, where each tuple contains the x and y coordinates of a corner of the first polygon.
        corners2: A list of tuples, where each tuple contains the x and y coordinates of a corner of the second polygon.

    Returns:
        A list of tuples representing the coordinates of the intersection polygon.
    """
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    intersection_poly = poly1.intersection(poly2)
    intersection_corners = list(intersection_poly.exterior.coords)[:-1]
    return intersection_corners


def get_rotated_rect_corners(center, width, height, angle):
    """
    Calculate the corner positions of a rectangular after rotation.

    Args:
        center: A tuple or list representing the (x, y) coordinates of the center of the rectangle.
        width: The width of the rectangle.
        height: The height of the rectangle.
        angle: The rotation angle of the rectangle in radians.

    Returns:
        A list of tuples representing the (x, y) coordinates of the corners of the rectangle.
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    corners = [(center[0] + width / 2 * cos_a - height / 2 * sin_a, center[1] + width / 2 * sin_a + height / 2 * cos_a),
               (center[0] - width / 2 * cos_a - height / 2 * sin_a, center[1] - width / 2 * sin_a + height / 2 * cos_a),
               (center[0] - width / 2 * cos_a + height / 2 * sin_a, center[1] - width / 2 * sin_a - height / 2 * cos_a),
               (center[0] + width / 2 * cos_a + height / 2 * sin_a, center[1] + width / 2 * sin_a - height / 2 * cos_a)]
    return corners


def check_collision(ori_GT_object_info_dict, sample_bbox_info):
    collision_info = {}
    for obj_name, info in ori_GT_object_info_dict.items():  # for each bbox, check
        collision_info[obj_name] = {}
        center_X = info[4]
        center_Y = info[5]
        size_x = info[8]  # length
        size_y = info[7]  # width
        angle = np.arctan2(info[11], info[10]) * 180 / pi
        corners = get_rotated_rect_corners(center=(center_X, center_Y), width=size_x * 2, height=size_y * 2,
                                           angle=angle)
        bbox_center_X, bbox_center_Y, bbox_size_x, bbox_size_y, bbox_rotation = sample_bbox_info
        bbox_corners = get_rotated_rect_corners(center=(bbox_center_X, bbox_center_Y), width=bbox_size_x * 2,
                                                height=bbox_size_y * 2, angle=bbox_rotation)
        collision_info[obj_name]['iou'] = iou_rotated_rectangles(corners, bbox_corners)
        collision_info[obj_name]['corners'] = corners
        collision_info['bbox_corners'] = bbox_corners

    return collision_info



def get_view_projection_matrices(cam):
    scene = bpy.context.scene
    render = scene.render
    aspect_ratio = render.resolution_x / render.resolution_y
    cam_data = cam.data

    # Create the view matrix
    mat_view = cam.matrix_world.inverted()

    # Create the projection matrix
    if cam_data.type == 'PERSP':
        # Perspective camera
        fovy = cam_data.angle_y
        near = cam_data.clip_start
        far = cam_data.clip_end
        top = near * (fovy / 2)
        right = top * aspect_ratio

        mat_proj = Matrix(((near / right, 0, 0, 0),
                           (0, near / top, 0, 0),
                           (0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)),
                           (0, 0, -1, 0)))

    else:  # cam_data.type == 'ORTHO'
        # Orthographic camera
        scale_x = cam_data.ortho_scale
        scale_y = cam_data.ortho_scale * aspect_ratio
        near = cam_data.clip_start
        far = cam_data.clip_end

        right = scale_x / 2
        top = scale_y / 2

        mat_proj = Matrix(((1 / right, 0, 0, 0),
                           (0, 1 / top, 0, 0),
                           (0, 0, -2 / (far - near), -(far + near) / (far - near)),
                           (0, 0, 0, 1)))

    return mat_view, mat_proj

def world_to_screen_coords(world_coord, cam):
    mat_view, mat_proj = get_view_projection_matrices(cam)
    scene = bpy.context.scene

    # Transform the world coordinate to camera and clip coordinates
    camera_coord = mat_view @ world_coord
    clip_coord = mat_proj @ camera_coord

    # Convert to normalized device coordinates (NDC)
    ndc_coord = clip_coord.to_3d() / clip_coord.to_3d().length

    # Convert NDC to screen coordinates taking aspect ratio and pixel aspect ratio into account
    aspect_ratio = scene.render.resolution_x / scene.render.resolution_y
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    screen_coord = (ndc_coord.xy * Vector((aspect_ratio, 1)) + Vector((1, 1))) * 0.5 * Vector(
        (scene.render.resolution_x * pixel_aspect_ratio, scene.render.resolution_y))

    return screen_coord

# Function to load an environment map and set it as the world background
def load_environment_map(filepath, rotation_degrees, intensity):
    # Load the image
    bpy.ops.image.open(filepath=filepath)

    # Get the image object
    env_map = bpy.data.images[os.path.basename(filepath)]

    # Create a new world if necessary
    if not bpy.data.worlds.get("Environment_World"):
        bpy.data.worlds.new("Environment_World")

    # Set the new world as the active world
    world = bpy.data.worlds["Environment_World"]
    bpy.context.scene.world = world

    # Use nodes for the world
    world.use_nodes = True

    # Clear existing nodes
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create the necessary nodes
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    background_node = nodes.new(type='ShaderNodeBackground')
    environment_texture_node = nodes.new(type='ShaderNodeTexEnvironment')
    mapping_node = nodes.new(type='ShaderNodeMapping')
    texture_coord_node = nodes.new(type='ShaderNodeTexCoord')

    # Set the environment texture
    environment_texture_node.image = env_map

    # Set rotation
#    mapping_node.rotation.z = rotation_degrees * 0.0174533  # Convert degrees to radians
    mapping_node.inputs[2].default_value[2]=rotation_degrees * 0.0174533  # Convert degrees to radians
    # control intensity of the environment map
    bpy.data.worlds["Environment_World"].node_tree.nodes["Background"].inputs[1].default_value = intensity

    # Connect the nodes
    links = world.node_tree.links
    links.new(texture_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], environment_texture_node.inputs['Vector'])
    links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
    links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

'''Main begin'''

def main(args):

    '''Prepare'''
    root_path = args.root_path
    # Load all train scene data
    with open(os.path.join(root_path, 'train_data_idx.txt')) as f:
        train_data_ids_raw = f.readlines()

    train_data_ids = ["{:06d}".format(int(id.split('\n')[0])) for id in train_data_ids_raw]  # change format
    # load objaverse pool for the interested classes
    objaverse_object_name_list = ["bed", "table", "sofa", "chair", "desk", "dresser", "nightstand", "bookshelf", "bathtub",
                                "toilet"]
    objaverse_objects_dict = {}

    with open('data/objaverse/objaverse.json') as f:
        objaverse_objects_dict = json.load(f)
    obj_root_path = args.obj_root_path


    # Insertion hyperparameters
    max_iter = args.max_iter # only insert one object for each scene
    random.seed(args.random_seed) # reproduce
    ilog = args.ilog # environmen map parameter
    istrength = args.istrength # environmen map parameter
    insertion_mode = args.insertion_mode
    out_folder_name = 'insertion_ilog{}_istren{}_{}'.format(ilog, istrength, insertion_mode) # good floor
    os.makedirs(os.path.join(root_path, out_folder_name, 'label'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'inserted_foreground'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'compositional_image'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'envmap'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'insert_object_log'), exist_ok=True)
    envmap_root = os.path.join(root_path, 'envmap')
 

    '''Go over each scene'''
    for iter in range(max_iter): # number of insertion times 
        for index, scene_id in enumerate(train_data_ids): 

            print('Processing scene id {}, index {}'.format(scene_id, index))

            # if there is no floor detected, skip this image
            plane_path = os.path.join(root_path, 'plane/{}/floor_plane_statistics_noceiling_threshold0.04.json'.format(scene_id))
            if not os.path.exists(plane_path):
                print('Skip (nofloor) image id ', scene_id)
                file1 = open(os.path.join(root_path, out_folder_name, 'nofloor_skip.txt'), "a")  # append mode
                file1.write("error {}  \n".format(scene_id))
                file1.close()
                continue
            if os.path.exists(os.path.join(root_path, out_folder_name, 'compositional_image', '{}_{}.png'.format(scene_id, iter))):  # already processed
                continue
            try:
                # open the blend file
                bpy.ops.wm.open_mainfile(filepath=os.path.join(root_path, 'insertion_template.blend'))
                # remove all except camera and point
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                # bpy.data.objects["Point"].select_set(False)
                bpy.ops.object.delete()
                '''
                1 Background operation
                '''

                # load the camera information
                # the txt file save column-wise matrix
                with open(os.path.join(root_path, 'calib', '{}.txt'.format(scene_id))) as f:
                    lines = f.readlines()

                # Extrinsic
                R_raw = np.array([float(ele) for ele in lines[0].split('\n')[0].split(' ')]).reshape((3, 3)).T  # (3,3)
                # transform R with blender convention
                flip_yz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                R = np.matmul(R_raw, flip_yz)
                matrix_world = np.eye(4)
                matrix_world[:3, :3] = R

                # change the camera Extrinsic parameters

                cam = bpy.data.objects['Camera']
                cam.matrix_world = Matrix(matrix_world)
                cam.location = Vector([0, 0, 0])

                # Intrinsic
                K = np.array([float(ele) for ele in lines[1].split('\n')[0].split(' ')]).reshape((3, 3)).T
                cx = K[0, 2]
                cy = K[1, 2]
                fx = K[0, 0]
                fy = K[1, 1]

                if fx != fy:
                    print('Warning: fx != fy')

                # change the camera Intrinsic parameters
                bpy.data.scenes['Scene'].render.resolution_x = cx * 2
                bpy.data.scenes['Scene'].render.resolution_y = cy * 2
                # obtain the focal lense and sensor width
                bpy.data.cameras[bpy.data.cameras.keys()[0]].sensor_width = 36  # fix the sensor_width
                alpha = bpy.data.scenes['Scene'].render.resolution_x / bpy.data.cameras[bpy.data.cameras.keys()[0]].sensor_width
                bpy.data.cameras[bpy.data.cameras.keys()[0]].lens = fx / alpha  # calculate the lens

                '''
                2 Insertion operation
                '''
                # Obtain plane and SUNRGBD object statistics information as reference

                '''Obtain plane information'''
                with open(plane_path, 'r') as fh:
                    plane_statistics_json = json.load(fh)

                floor_center_X = plane_statistics_json['floor']['mean'][0]
                floor_center_Y = plane_statistics_json['floor']['mean'][1]
                floor_center_Z = plane_statistics_json['floor']['mean'][2]
                floor_std_X = plane_statistics_json['floor']['std'][0]
                floor_std_Y = plane_statistics_json['floor']['std'][1]
                floor_std_Z = plane_statistics_json['floor']['std'][2]
                floor_size = plane_statistics_json['floor']['size']  # 's', 'm', 'l'
                with open(os.path.join(root_path, 'SUNRGBD_objects_statistic.json'), 'r') as fh:
                    insertion_size_statistics = json.load(fh)

                '''Obtain original GT objects information'''
                with open(os.path.join(root_path, 'label', '{}.txt'.format(scene_id))) as f:
                    lines = f.readlines()

                ori_GT_object_info_dict = {}
                ori_GT_object_info_list = []
                ori_cate_list = []
                for id, line in enumerate(lines):  # for each GT object
                    raw_info_list = [line.split('\n')[0].split(' ')][0]
                    class_name = raw_info_list[0]
                    if class_name not in ori_cate_list:
                        if class_name == 'night_stand':
                            ori_cate_list.append('nightstand')
                        else:
                            ori_cate_list.append(class_name)
                    info_list = [float(ele) for ele in raw_info_list[1:]]
                    ori_GT_object_info_dict[str(id) + '_' + class_name] = info_list  # load all info as dict
                    ori_GT_object_info_list.append(info_list)  # easy to compute min max

                # random select one object class or select from existing classes
                insert_class = select_inserted_object_class(objaverse_object_name_list, ori_cate_list,
                                            args.insertion_mode) 
                # random select one object form objaverse and insert
                inserted_object_path = select_inserted_object(insert_class)
                insert_log_dict = {}
                insert_log_dict['insert_class'] = insert_class
                insert_log_dict['inserted_object_path'] = inserted_object_path  # document the inserted object path
                inobject, bbox = load_insert_object(filepath=inserted_object_path)  # random choose one object from pool

                # basic parameter of insert object
                size_z = random.gauss(mu=insertion_size_statistics[insert_class][0],
                                    sigma=insertion_size_statistics[insert_class][1])  # mean and std
                insert_log_dict['size_z'] = size_z
                reference_object_height = size_z * 2  # size *2 = dimension
                obj_shrink_factor = bbox.dimensions[2] / reference_object_height  # height of inserted / height of reference

                '''collision check'''
                best_parameter = {}
                for i in range(1000):  # do 1000 times sample and collision check
                    sample_scale = random.uniform(1, args.resize_factor)  # only become smaller
                    sample_position_X = random.uniform(floor_center_X - floor_std_X,
                                                    floor_center_X + floor_std_X)  # randomly sample from a uniform distribution
                    sample_position_Y = random.uniform(floor_center_Y - floor_std_Y, floor_center_Y + floor_std_Y)
                    sample_rotation = random.uniform(-180, 180)  # randomly sample good pose

                    if i == 0:  # first iteration
                        best_parameter['iou'] = 1  # maximum
                        best_parameter['sample_scale'] = sample_scale
                        best_parameter['sample_position_X'] = sample_position_X
                        best_parameter['sample_position_Y'] = sample_position_Y
                        best_parameter['sample_rotation'] = sample_rotation
                    sample_obj_shrink_factor = obj_shrink_factor * sample_scale  # shrink more
                    sample_inobject_location = Vector([sample_position_X, sample_position_Y, floor_center_Z])
                    sample_bbox_location = sample_inobject_location
                    sample_bbox_size = [bbox.dimensions[0] / sample_obj_shrink_factor / 2,
                                        bbox.dimensions[1] / sample_obj_shrink_factor / 2]

                    sample_bbox_info = [sample_bbox_location[0], sample_bbox_location[1], sample_bbox_size[0],
                                        sample_bbox_size[1], sample_rotation]

                    bbox_center_X = bbox.location[0]
                    bbox_center_Y = bbox.location[1]
                    bbox_size_x = bbox.dimensions[0] / obj_shrink_factor / 2  # need to devide the obj_shrink_factor first
                    bbox_size_y = bbox.dimensions[1] / obj_shrink_factor / 2

                    # collision check
                    collision_info = check_collision(ori_GT_object_info_dict, sample_bbox_info)
                    iou = 0.0
                    for obj in collision_info.keys():  # for each object, check if it is satisfied
                        if obj != 'bbox_corners':  # valid object
                            iou += collision_info[obj]['iou']
                    #    print(i, iou)
                    if iou <= 0.01:  # satisfied the requirements
                        print('satisfied after {} iteration'.format(i))
                        best_parameter['iou'] = iou
                        best_parameter['sample_scale'] = sample_scale
                        best_parameter['sample_position_X'] = sample_position_X
                        best_parameter['sample_position_Y'] = sample_position_Y
                        best_parameter['sample_rotation'] = sample_rotation
                        print('no collision after {} iteration'.format(i))
                        break

                    elif iou < best_parameter['iou']:  # find a better position
                        best_parameter['iou'] = iou
                        best_parameter['sample_scale'] = sample_scale
                        best_parameter['sample_position_X'] = sample_position_X
                        best_parameter['sample_position_Y'] = sample_position_Y
                        best_parameter['sample_rotation'] = sample_rotation
                        best_parameter['iterations'] = i
                # save insertion information for reproduce
                insert_log_dict['best_parameter'] = best_parameter
                with open(os.path.join(root_path, out_folder_name, 'insert_object_log', '{}_{}.json'.format(scene_id, iter)), "w") as outfile:
                    # Write the dictionary to the file in JSON format
                    json.dump(insert_log_dict, outfile, indent=4)
                # final resize after the collision check
                inobject.scale /= obj_shrink_factor * best_parameter['sample_scale']
                bbox.scale /= obj_shrink_factor * best_parameter['sample_scale']
                inobject.rotation_mode = 'XYZ'
                inobject.rotation_euler[2] += best_parameter['sample_rotation'] / 180 * pi
                bbox.rotation_euler[2] += best_parameter['sample_rotation'] / 180 * pi

                ##move position
                inobject.location[0] = best_parameter['sample_position_X']
                inobject.location[1] = best_parameter['sample_position_Y']
                inobject.location[2] = floor_center_Z + bbox.dimensions[2] / obj_shrink_factor / best_parameter[
                    'sample_scale'] / 2  # put the bottom on the ground
                bbox.location = inobject.location

                insert_3D_info = {}
                final_shrink_factor = obj_shrink_factor * best_parameter['sample_scale']
                insert_3D_info['class'] = insert_class
                insert_3D_info['centroid_X'] = bbox.location[0]
                insert_3D_info['centroid_Y'] = bbox.location[1]
                insert_3D_info['centroid_Z'] = bbox.location[2]
                # SUN RGBD data[9] is x_size (length), data[8] is y_size (width), data[10] is z_size (height) in our depth coordinate system,
                insert_3D_info['width'] = bbox.dimensions[1] / 2 / final_shrink_factor
                insert_3D_info['length'] = bbox.dimensions[0] / 2 / final_shrink_factor
                insert_3D_info['height'] = bbox.dimensions[2] / 2 / final_shrink_factor
                insert_3D_info['heading_angle'] = best_parameter[
                                                    'sample_rotation'] / 180 * pi  # yaw angle = roration_z (in our depth coordinate) = 0

                # insert_3D_info['size'] = np.array([data[9], data[8], data[10]]) * 2
                insert_3D_info['size'] = list(
                    [bbox.dimensions[0] / final_shrink_factor, bbox.dimensions[1] / final_shrink_factor,
                    bbox.dimensions[2] / final_shrink_factor])

                json_object = json.dumps(insert_3D_info, indent=4)

                # Writing to sample.json
                with open(os.path.join(root_path, out_folder_name, 'label', '{}_{}.json'.format(scene_id, iter)), "w") as outfile:
                    outfile.write(json_object)

                # add plane as shadow caster
                plane_location = bbox.location.copy()
                plane_location[2] -= bbox.dimensions[2] / 2 / final_shrink_factor
                bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=plane_location)
                plane = bpy.context.active_object
                # plane.cycles.is_shadow_catcher = True
                bpy.data.objects["Plane"].is_shadow_catcher = True
                
                '''
                3 Add dynamic illumination
                '''

                '''obtain 2D position given an 3D pixel'''

                backimg = mpimg.imread(os.path.join(root_path, 'SUNRGBD_data/image', '{}.jpg'.format(scene_id)))

                # Calculate the object's center in world coordinates (replace this with the actual object center)
                object_center_world = Vector((inobject.location[0], inobject.location[1], floor_center_Z))

                # Get the 2D position of the object's center in the rendered image
                object_center_2d = world_to_screen_coords(object_center_world, cam)

                # The origin of the coordinate system in Blender is at the bottom-left corner, while the origin in matplotlib is at the top-left corner. Therefore, you need to adjust the y-coordinate of the bounding box when drawing it with matplotlib.
                # Draw the object's center as a red dot on the image
                object_center_2d_img = Vector((object_center_2d.x, backimg.shape[0] - object_center_2d.y))
                if object_center_2d_img[0] < 0: # width
                    object_center_2d_img[0] = 0
                elif object_center_2d_img[0] > backimg.shape[1]:
                    object_center_2d_img[0] = backimg.shape[1]

                if object_center_2d_img[1] < 0: # height
                    object_center_2d_img[1] = 0
                elif object_center_2d_img[1] > backimg.shape[0]:
                    object_center_2d_img[1] = backimg.shape[0]
                '''Load the dynamic environment map'''

                # calculate which specific 8x8 environment map should we use.
                envCol = 160
                envRow = 120
                nh = backimg.shape[0]  # 530, height
                nw = backimg.shape[1]  # 730, width

                if nh < nw:  # hight is small
                    newW = envCol  # fix the width first: 160
                    newH = int(float(newW) / float(nw) * nh)  # same zoom ratio
                zoom_ratio = float(nw) / float(newW)  # how many pixel in raw image are treated as one environment map

                envmap_index = Vector(
                    (object_center_2d_img[1] / zoom_ratio, object_center_2d_img[0] / zoom_ratio))  # first row, then column

                loaded_env_data = np.load(os.path.join(envmap_root, '{}_envmap1.png.npz'.format(scene_id)))

                # Create a NumPy array (this should be a valid HDR image data)
                row = int(envmap_index[0])
                column = int(envmap_index[1])

                if row > loaded_env_data['env'].shape[0]-1: # > 115
                    row = loaded_env_data['env'].shape[0]-1
                if column > loaded_env_data['env'].shape[1]-1: # > 159
                    column = loaded_env_data['env'].shape[1]-1
                array = loaded_env_data['env'][row, column]
                fill_zero_down_array = np.concatenate((array, np.zeros_like(array)), axis=0)  # correct filling
                fill_zero_down_array_log = fill_zero_down_array ** ilog
                # Save the NumPy array to an .hdr file
                env_map_filepath = os.path.join(root_path, out_folder_name, 'envmap','{}_{}{}_filldown.hdr'.format(scene_id, row, column))
                cv2.imwrite(env_map_filepath, fill_zero_down_array_log)

                # Set your environment map file path and rotation degrees
                rotation_degrees = 180  # Replace with your desired rotation in degrees

                # Call the function to load and rotate the environment map
                load_environment_map(env_map_filepath, rotation_degrees, istrength)


                '''
                3 Render and save operation
                '''

                bpy.context.scene.render.filepath = os.path.join(root_path, out_folder_name, 'inserted_foreground',
                                                                '{}_{}.png'.format(scene_id, iter))

                bpy.ops.render.render(write_still=True)  # here save RGBA image

                ## read the saved inserted object image

                # img = mpimg.imread('/Users/yunhaoge/000063_render.png')

                # Front Image
                filename = os.path.join(root_path, out_folder_name, 'inserted_foreground', '{}_{}.png'.format(scene_id, iter))

                # Back Image
                filename1 = os.path.join(root_path, 'image', '{}.jpg'.format(scene_id))

                # Open Front Image
                frontImage = Image.open(filename)

                # Open Background Image
                background = Image.open(filename1)

                # Convert image to RGBA, already in RGBA
                frontImage = frontImage.convert("RGBA")

                # Convert image to RGBA
                frontImage = frontImage.convert("RGBA")

                # Paste the frontImage at (0, 0)
                background.paste(frontImage, (0, 0), frontImage)

                # Save this image
                background.save(os.path.join(root_path, out_folder_name, 'compositional_image', '{}_{}.png'.format(scene_id, iter)),
                                format="png")

                '''
                remove current bbox, inserted object, and axis
                '''

                # remove all except camera and point
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                # bpy.data.objects["Point"].select_set(False)
                bpy.ops.object.delete()

            except:

                # remove all except camera and point
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                # bpy.data.objects["Point"].select_set(False)
                bpy.ops.object.delete()

                file1 = open(os.path.join(root_path, out_folder_name, 'error_log.txt'), "a")  # append mode
                file1.write("error {}  \n".format(scene_id))
                file1.close()

if __name__ == '__main__':
    '''noise data'''
    parser = argparse.ArgumentParser(description='3d_copy_paste')
    
    parser.add_argument('--root_path', default="data/sunrgbd/sunrgbd_trainval/", help='root path for SUN RGB-D data')
    parser.add_argument('--obj_root_path', default="data/objaverse/obj", help='root path for Objaverse data')
    parser.add_argument('--insertion_mode', default="context", help='random: randomly insert any objects given scene,  context: only insert existing category objects in the original scene during insertion.')
    parser.add_argument('--max_iter', type=int, default=1, help='number of inserted objects for each scene')
    parser.add_argument('--random_seed', type=int, default=1, help='random seed for reproduce')
    parser.add_argument('--resize_factor', type=int, default=3, help='to shrink the size of the inserted object to handle inserting a large object in a small empty floor scenario')
    parser.add_argument('--ilog', type=int, default=2, help='envirnment map parameter')
    parser.add_argument('--istrength', type=int, default=2, help='envirnment map parameter')
    args = parser.parse_args()

    main(args)