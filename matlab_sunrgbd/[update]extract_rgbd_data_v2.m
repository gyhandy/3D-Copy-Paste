%% Save extra pointcloud, raw_depth, xyz_depth, rgb_raw than raw extract_rgbd_ata_v2.m file

% Copyright (c) Facebook, Inc. and its affiliates.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

%% Dump SUNRGBD data to our format
% for each sample, we have RGB image, 2d boxes.
% point cloud (in camera coordinate), calibration and 3d boxes.
%
% Compared to extract_rgbd_data.m in frustum_pointents, use v2 2D and 3D
% bboxes.
%
% Author: Charles R. Qi
%
clear; close all; clc;
% addpath(genpath('.'))
% addpath('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/readData')
addpath('/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDtoolbox/readData')
%% V1 2D&3D BB and Seg masks
% load('./Metadata/SUNRGBDMeta.mat')
% load('./Metadata/SUNRGBD2Dseg.mat')

%% V2 3DBB annotations (overwrites SUNRGBDMeta)
load('/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat');
load('/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat');
%% Create folders
depth_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/depth/';
image_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/image/';
calib_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/calib/';
det_label_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/label/';
seg_label_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/seg_label/';
pointcloud_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/pointcloud/'; % new pointcloud
raw_depth_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/raw_depth/'; 
xyz_depth_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/xyz_depth/'; % new xyz_depth
rgb_raw_folder = '/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/rgb_raw/'; % new xyz_depth

mkdir(depth_folder);
mkdir(image_folder);
mkdir(calib_folder);
mkdir(det_label_folder);
mkdir(seg_label_folder);
mkdir(pointcloud_folder);
mkdir(raw_depth_folder);
mkdir(xyz_depth_folder);
mkdir(rgb_raw_folder);
%% Read
parfor imageId = 1:10335
    imageId
try
% imageId = 63; %63, 2302;10335
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/OFFICIAL_SUNRGBD', data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/OFFICIAL_SUNRGBD', data.rgbpath);

copyfile(data.depthpath, sprintf('%s/%06d.png', raw_depth_folder, imageId));


% Write point cloud in depth map
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
parsave(strcat(xyz_depth_folder, mat_filename), points3d);
parsave(strcat(rgb_raw_folder, mat_filename), rgb);

rgb(isnan(points3d(:,1)),:) = [];
points3d(isnan(points3d(:,1)),:) = [];
points3d_rgb = [points3d, rgb];

% Build Point cloud and visualize
pc_xyz_rgb = pointCloud(points3d); % create pointCloud format data structure, with only xyz information
pc_xyz_rgb.Color = rgb; % add color (rgb) information
pc_filename = strcat(num2str(imageId,'%06d'), '.ply');
% pcshow(pc_xyz_rgb) % visualization
pcwrite(pc_xyz_rgb, strcat(pointcloud_folder, pc_filename));

% MAT files are 3x smaller than TXT files. In Python we can use
% scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
parsave(strcat(depth_folder, mat_filename), points3d_rgb);

% Write images
copyfile(data.rgbpath, sprintf('%s/%06d.jpg', image_folder, imageId));

% Write calibration
dlmwrite(strcat(calib_folder, txt_filename), data.Rtilt(:)', 'delimiter', ' ');
dlmwrite(strcat(calib_folder, txt_filename), data.K(:)', 'delimiter', ' ', '-append');

% Write 2D and 3D box label
data2d = SUNRGBDMeta2DBB(imageId);
fid = fopen(strcat(det_label_folder, txt_filename), 'w');
for j = 1:length(data.groundtruth3DBB)
    centroid = data.groundtruth3DBB(j).centroid;
    classname = data.groundtruth3DBB(j).classname;
    orientation = data.groundtruth3DBB(j).orientation;
    coeffs = abs(data.groundtruth3DBB(j).coeffs);
    box2d = data2d.groundtruth2DBB(j).gtBb2D;
    assert(strcmp(data2d.groundtruth2DBB(j).classname, classname));
    fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
end
fclose(fid);

catch
end

end

function parsave(filename, instance)
save(filename, 'instance');
end
