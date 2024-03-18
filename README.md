# 3D-Copy-Paste
[NeurIPS 2023] 3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection
### [Project Page](https://gyhandy.github.io/3D-Copy-Paste/) | [arXiv](https://arxiv.org/abs/2312.05277) | [Paper](https://arxiv.org/pdf/2312.05277.pdf)

<div align="center">
    <img src="./static/images/3D-copy-paste.gif" alt="Editor" width="1200">
</div>



## Detailed steps.

Overview: the following steps conduct 3D Copy Paste by inserting [Objaverse](https://objaverse.allenai.org/) objects into [SUN RGB-D](https://rgbd.cs.princeton.edu/) dataset. The insertion and rendering use [Blender](https://www.blender.org/). Then train an [ImvoxelNet](https://github.com/SamsungLabs/imvoxelnet) model for monocular 3D object detection using [MMdetection3D](https://github.com/open-mmlab/mmdetection3d) code. 


### Step1 Download inserted objects and RGBD scene

#### 1.1 Download inserted objects
Option 1:

(1) Download a selected [Objaverse](https://objaverse.allenai.org/) subsets that contains 10-class objects belongs to SUNRGBD, based on
a Objaverse id list: [objaverse.json](https://downloads.cs.stanford.edu/viscam/3DCopyPaste/objaverse.json) (detailed statistics in paper Table 1). 

(2) Transfer raw '.glb' format to '.obj' format for easier loading and process in [Blender](https://www.blender.org/) during insertion.
```bash
python export_glb_2_obj.py
```
all Objaverse raw objects saved into a folder '/data/objaverse/obj'

Option 2:

Directly download processed objaverse objects in '.obj' format and 'objaverse.json' from [here](https://downloads.cs.stanford.edu/viscam/3DCopyPaste/objaverse_objects.zip)

After downloadeing or process, the data should follow the following structure.

```
data
├── objaverse
│   ├── obj # folder that contains all .obj folders for each object
│   ├── objaverse.json

```


#### 1.2 Download and preprocess of RGBD scene
Download and process [SUN RGB-D](https://rgbd.cs.princeton.edu/) dataset following the instructions from mmdetection3d [here](https://github.com/open-mmlab/mmdetection3d/tree/main/data/sunrgbd).

Note: during process, substitute the original 'extract_rgbd_data_v2.m' with '/matlab_sunrgbd/[update]extract_rgbd_data_v2.m', which will save extra pointcloud, raw_depth, xyz_depth, rgb_raw than the raw extract_rgbd_ata_v2.m file. The processed data is
useful for the following insertion. [SUN RGB-D data structure](https://mmdetection3d.readthedocs.io/en/v0.17.1/datasets/sunrgbd_det.html) understanding is helpful for data processing and how to convert the dataset for final monocular 3D object detection model training using [MMdetection3D](https://github.com/open-mmlab/mmdetection3d).

After process, the dataset structure should be as below. Note: The 'sunrgbd_trainval' folder contains more contents than the folder that following original mmdetection3d instructions [here](https://github.com/open-mmlab/mmdetection3d/tree/main/data/sunrgbd). 

```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   ├── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   ├── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt
│   ├── val_data_idx.txt
│   ├── pointcloud 
│   ├── raw_depth 
│   ├── xyz_depth 
│   ├── rgb_raw
├── points
├── sunrgbd_infos_train.pkl
├── sunrgbd_infos_val.pkl

```

### Step2 Physically plausible position, pose and size
#### Plane detection on RGBD scene
(1) Download the RGBDPlaneDetection code
[RGBDPlaneDetection](https://github.com/chaowang15/RGBDPlaneDetection)

(2) Resize both RGB images and depth from SUN RGB-D to 640x480 (raw RGB image size is 730x530). The reason of resize is because RGBDPlaneDetection use 640x480 as default image and depth size.
```bash
python plane_detection_1_resize_to_640480.py
```

(3) Run plane detection and reconstruction for all SUN RGB-D training data, given RGB and depth as input
```bash
cd RGBDPlaneDetection
bash run_RGBDPlaneDetection.sh
```

(4) Calculate statistics of each reconstructed plane, save mean, std, max, min for following horizontal plane selection later.
```bash
python plane_detection_2_plane_analysis.py
```

(5) Based on the plane statistics, find the horizontal plane and floor.
```bash
python plane_detection_3_floor_detection_and_filter.py
```

### Step3 Physically plausible lighting
#### Estimate environment map for each RGBD scene
Option 1:

Clone and build [InverseRenderingOfIndoorScene](https://github.com/lzqsd/InverseRenderingOfIndoorScene) 

```bash
bash runRealSUNRGBD_train.sh
```
Note:

(1)'--dataRoot' a image folder that contains all interested images you want to run environment map estimation, here is SUN RGB-D images

(2)'--imList' a txt file to store the image name (e.g., 007503.png, 007504.png...).

It will generate envmap in 'RealSUNRGBD/envmap'. 

Option 2: 

Directly download the processed envmap from [here](https://downloads.cs.stanford.edu/viscam/3DCopyPaste/envmap.zip)

Note: For easier management, you may move the envmap folder under 'sunrgbd_trainval' folder.

### Step4 Object insertion
#### 4.1 Prepare all dataset material in folder "sunrgbd_trainval"

In step 1.2, we already preprocess the SUN RGB-D dataset, we also add plane reconstrution (step2) and lighting estimation results (step3). 

prepare "SUNRGBD_objects_statistic.json" which save the mean and std of object hight for each interest class 
```bash
python SUNRGBD_objects_statistic.py
```

Prepare a Blender template "insertion_template.blend" which can help generate the shadows and other predefined rendering parameters (you can download it from [here](https://downloads.cs.stanford.edu/viscam/3DCopyPaste/insertion_template.blend))

After all above processes, move all above materials under folder 'sunrgbd_trainval'. It should has a following data structure. (# We add some comments for some important folder that we will use during the insertion.)
```
sunrgbd
├── README.md
├── matlab
│   ├── extract_rgbd_data_v1.m
│   ├── extract_rgbd_data_v2.m
│   ├── extract_split.m
├── OFFICIAL_SUNRGBD
│   ├── SUNRGBD
│   ├── SUNRGBDMeta2DBB_v2.mat
│   ├── SUNRGBDMeta3DBB_v2.mat
│   ├── SUNRGBDtoolbox
├── sunrgbd_trainval
│   ├── calib # save the camera extrinsic and intrinsic for rendering the inserted object. We use is during Blender rendering.
│   ├── depth
│   ├── image
│   ├── label
│   ├── label_v1
│   ├── seg_label
│   ├── train_data_idx.txt  # save the 4 digit SUNRGBD image id e.g., "5051"
│   ├── val_data_idx.txt
│   ├── pointcloud  # save the scene point cloud data
│   ├── raw_depth 
│   ├── xyz_depth 
│   ├── rgb_raw
│   ├── envmap  # save the environment map data in step3
│   ├── plane  # save the reconstructed plane information in step2
│   ├── object_statistics  # save the mean and std of object hight for each interest class
│   ├── SUNRGBD_objects_statistic.json  # save the mean and std of object hight for each interest class
│   ├── insertion_template.blend  # save the mean and std of object hight for each interest class
├── points
├── sunrgbd_infos_train.pkl
├── sunrgbd_infos_val.pkl

```
#### 4.2 Conduct insertion
First render inserted object images with groundtruth 3D bounding box, then paste it on the original scene images, generate the final images. 
```bash
python 3d_copy_paste.py
``` 


### Step5 Create training dataset for monocular object detection: ImvoxelNet 
#### Create dataset for training ImvoxelNet

(1) Git clone [MMdetection3D](https://github.com/open-mmlab/mmdetection3d) code.

(2) Substitute two data creation files in our imvoxelnet folder to mmdetection3d/tools/dataset_convers:
'indoor_converter.py', 'sunrgbd_data_utils.py'

Copy one data creation files in our imvoxelnet folder to mmdetection3d/tools: 
'create_imvoxelnet_dataset.py'

(3) ImvoxelNet dataset creation with 3D Copy Paste augmentation.
```bash
python create_imvoxelnet_dataset.py sunrgbd --root-path ./data/sunrgbd --out-dir ./data/insertion_ilog2_istren2_context 
 --extra-tag sunrgbd --insert_set insert_img/insert/insertion_ilog2_istren2_context --index_txt insert_only_train_data_idx.txt --workers 32
``` 

### Step6 Train ImvoxelNet

Copy training config file in our imvoxelnet folder to mmdetection3d/configs/imvoxelnet: 
'config_insertion_ilog2_istren2_context.py'

```bash
bash tools/dist_train.sh \
  configs/imvoxelnet/config_insertion_ilog2_istren2_context.py 2
``` 