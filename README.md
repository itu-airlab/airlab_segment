airlab_segment
============

This is an ROS package implementation that is composed of Euclidean Clustering and Region Growing algorithms in PointCloud Library.

The node can be used as a standalone 3D segmentation algorithm or as an data source to [Violet](https://github.com/itu-airlab/violet)

# License
Violet is licensed with GPL Version 3 or later.

# Compilation instructions

To compile `airlab_segment` package, you can clone this repository directly under your catkin workspace.

## Compile time build options

### Location postfiltering

`-DENABLE_LOCATION_POSTFILTERING` CMake switch enables/disables building location postfiltering for defining a workspace for segmentation results. It's by default enabled.


# Nodes

## euclidean_segmenter

## Parameters
* **`input_pc_topic`** : String
     The input point cloud topic
     Default: `camera/depth_registered/points`

* **`world_tf_frame`** : String
    The destination frame for published segments.
    Default: `world`


* **`camera_tf_frame`** : String
    The optical frame in which camera message published.
    Default: `camera_rgb_optical_frame`

* **`register_to_violet`** : Bool
    Whether register to Violet as an input source
    Default: `true`
    
* **`violet_namespace`** : String
    Violet namespace that used in registering this node as an input source
    Default: `violet`

* **`vis_pc_publish_topic`** : String
    The topic name for publishing visualization cloud
    Default: `segmented_cloud`
    