cd /Users/yunhaoge/PycharmProjects/RGBDPlaneDetection
target_path='/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/plane'
for i in $(seq 1 1 10335)
do
  mkdir $target_path/$(printf "%06d" i)
  echo "Detecting plane on scene ${i}"
  rgb=/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/image_resize_640480/$(printf "%06d" i).jpg
  echo $rgb
  depth=/Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/raw_depth_resize_640480/$(printf "%06d" i).png
  echo $depth
  ./build/RGBDPlaneDetection -o /Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/image_resize_640480/$(printf "%06d" i).jpg /Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/raw_depth_resize_640480/$(printf "%06d" i).png /Users/yunhaoge/PycharmProjects/mmdetection3d/data/sunrgbd/sunrgbd_trainval/plane/$(printf "%06d" i)/
done