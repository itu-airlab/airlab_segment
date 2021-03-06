/*
 *  Copyright (C) 2018 Ongun Kanat <ongun.kanat@gmail.com>
 *  Copyright (C) 2018 Istanbul Technical University
 *                     Artificial Intelligence and Robotics Laboratory
 *                     <air.cs.itu.edu.tr>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <cstdio>
#include <cstring>
#include <cmath>
/* Boost */
#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
/* ROS */
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <dynamic_reconfigure/server.h>
/* PCL */
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
/* PCL - ROS integration */
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
/* Own Libs */
#include <airlab_segment/SegmentParamsConfig.h>
#include <violet_msgs/DetectionInfo.h>
#include <violet_srvs/RegisterSource.h>

namespace {
/* Definitions */
typedef pcl::PointXYZRGBA PointType; // NEVER USE PointXYZRGB PCL shits on his pants
typedef pcl::PointCloud<PointType> CloudType;

template <typename T>
static inline void emptyDestructorForStack(const T*) { return; }

namespace parameters {
/* Topic Parameters */
std::string input_pc_topic; // PointCloud input topic
std::string vis_pc_publish_topic; // Visualization PointCloud topic
std::string violet_publish_topic; // Violet detection info topic

/* Calculation Parameters */
float min_depth; // Passthru min depth
float max_depth; // Passthru max depth
int plane_min_inliers; // Minimum number of inlier points in the plane
int cluster_max_inliers; // Maximum number of elements in an euclidean segment
int cluster_min_inliers; // Minimum number of elements in an euclidean segment
float euclidean_distance_threshold; // The maximum distance threshold for including a point to an eucl. segment

int region_growing_min_inliers; // Minimum number of elements of a segment extracted via reg. grow
int region_growing_point_color_thresh; //The maximum color distance for including a point into a region
int region_growing_region_color_thresh; // The maximum color distance for merging a region with another
float region_growing_distance_threshold; // The max distance threshold for including a point to a grown region

std::string camera_tf_frame; // TF frame for camera
std::string world_tf_frame; // TF frame for world, which point clouds transformed to

int thread_count; // Number of spinning threads
int queue_size;  // Point cloud queue size
}

ros::Publisher vis_cloud_pub; // Randomly colored visualization cloud publisher
ros::Publisher segmented_violet_obj_pub; // Violet object publisher

bool registered_to_violet; // Denotes if the node is registered to violet

static inline void distancePassThruFilter(const CloudType::ConstPtr input, CloudType::Ptr output);

static inline void segmentPlanesToExclude(const CloudType::ConstPtr input,
                            pcl::PointCloud<pcl::Label>::Ptr plane_labels,
                            std::vector<pcl::PointIndices> &plane_label_indices);

static inline void euclideanClusterCloud(const CloudType::ConstPtr input,
                           pcl::PointCloud<pcl::Label>::Ptr plane_labels,
                           const std::vector<pcl::PointIndices> &plane_label_indices,
                           pcl::PointCloud<pcl::Label> &euclidean_labels,
                           std::vector<pcl::PointIndices> &euclidean_label_indices);


static inline void colorSegmentation(const CloudType::ConstPtr input,
                                     const pcl::PointIndices &roi_indices,
                                     std::vector<pcl::PointIndices> &color_segment_indices);

static inline void colorizeCloud(CloudType& cloud);

static inline bool transformPointCloud(const CloudType &input, CloudType &output);

static inline void publishSegments(const CloudType::ConstPtr cloud, const std::vector<pcl::PointIndices> extracted_segment_indices);

void processPointCloud(const CloudType::ConstPtr &input)
{
    CloudType process_cloud;
    CloudType::Ptr cloud(&process_cloud, emptyDestructorForStack<CloudType>);
    distancePassThruFilter(input, cloud);

    pcl::PointCloud<pcl::Label>::Ptr plane_labels (new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> plane_label_indices;
    segmentPlanesToExclude(cloud, plane_labels, plane_label_indices);

    pcl::PointCloud<pcl::Label> euclidean_labels;
    std::vector<pcl::PointIndices> euclidean_label_indices;
    euclideanClusterCloud(cloud, plane_labels, plane_label_indices, euclidean_labels, euclidean_label_indices);
    size_t num_euclidean_segments = euclidean_label_indices.size();

    std::vector<pcl::PointIndices> extracted_segment_indices;

    for(size_t i_eucl = 0; i_eucl < num_euclidean_segments; ++i_eucl) {
        if(euclidean_label_indices[i_eucl].indices.size() > parameters::cluster_max_inliers ||
           euclidean_label_indices[i_eucl].indices.size() < parameters::cluster_min_inliers) {
            continue;
        }

        std::vector<pcl::PointIndices> color_segmented_indices;
        colorSegmentation(cloud, euclidean_label_indices[i_eucl], color_segmented_indices);
        size_t num_color_segments = color_segmented_indices.size();

        bool color_segments_too_small = (color_segmented_indices.size() == 0);
        for (size_t i_color = 0; i_color < num_color_segments; ++i_color) {
            // If there are too few points in any region growing segment reject all (for now)
            if(color_segmented_indices[i_color].indices.size() < parameters::region_growing_min_inliers) {
                color_segments_too_small = true;
                break;
            }
        }

        if(color_segments_too_small) {
            extracted_segment_indices.push_back(euclidean_label_indices[i_eucl]);
        }
        else {
            for (size_t i_color = 0; i_color < num_color_segments; ++i_color) {
                extracted_segment_indices.push_back(color_segmented_indices[i_color]);
            }
        }
    }

    publishSegments(cloud, extracted_segment_indices);
}

static inline void distancePassThruFilter(const CloudType::ConstPtr input, CloudType::Ptr output)
{
    static thread_local pcl::PassThrough<PointType> pass;
    pass.setFilterLimits(parameters::min_depth, parameters::max_depth);
    pass.setFilterFieldName("z");
    pass.setInputCloud(input);
    pass.setKeepOrganized(true);
    pass.filter(*output);
}

static inline void segmentPlanesToExclude(const CloudType::ConstPtr input,
                                          pcl::PointCloud<pcl::Label>::Ptr plane_labels,
                                          std::vector<pcl::PointIndices> &plane_label_indices)
{
    static thread_local pcl::IntegralImageNormalEstimation<PointType, pcl::Normal> normal_extractor;
    pcl::PointCloud<pcl::Normal> input_normals_cloud;
    pcl::PointCloud<pcl::Normal>::Ptr input_normals_ptr (&input_normals_cloud, emptyDestructorForStack<pcl::PointCloud<pcl::Normal> >);
    normal_extractor.setInputCloud(input);
    normal_extractor.setNormalEstimationMethod(normal_extractor.COVARIANCE_MATRIX);
    normal_extractor.setMaxDepthChangeFactor(0.02f);
    normal_extractor.setNormalSmoothingSize(20.0f);
    normal_extractor.compute(*input_normals_ptr);

    std::vector<pcl::PlanarRegion<PointType>, Eigen::aligned_allocator<pcl::PlanarRegion<PointType> > > planar_regions;
    std::vector<pcl::ModelCoefficients> plane_coefficients;
    std::vector<pcl::PointIndices> plane_point_indices;
    std::vector<pcl::PointIndices> boundary_indices;

    static thread_local pcl::OrganizedMultiPlaneSegmentation<PointType, pcl::Normal, pcl::Label> multi_plane_segmenter;

    multi_plane_segmenter.setDistanceThreshold(0.03);
    multi_plane_segmenter.setInputCloud(input);
    multi_plane_segmenter.setInputNormals(input_normals_ptr);

    multi_plane_segmenter.segmentAndRefine(planar_regions,
                                           plane_coefficients,
                                           plane_point_indices,
                                           plane_labels,
                                           plane_label_indices,
                                           boundary_indices);

}

void euclideanClusterCloud(const CloudType::ConstPtr input,
                           pcl::PointCloud<pcl::Label>::Ptr plane_labels,
                           const std::vector<pcl::PointIndices> &plane_label_indices,
                           pcl::PointCloud<pcl::Label> &euclidean_labels,
                           std::vector<pcl::PointIndices> &euclidean_label_indices)
{
    /* Create excluded indice vector for planes which are detected by MPS */
    std::vector<bool> excluded_indices(plane_label_indices.size(), false);
    size_t n_indices = plane_label_indices.size();
    // for each indice vector of each found plane by multi plane segmenter
    for (size_t i = 0; i < n_indices ; i++) {
        // if the current plane has greater number of points (indices) than `plane_min_inliers`
        // set exclusion true
        size_t sz = plane_label_indices[i].indices.size();
        excluded_indices[i] = sz > parameters::plane_min_inliers;
    }

    /* This is the compare functor for comparing two points(?) and its compare
     * function returns true when two points are close to each other smaller than a
     * radius
     */
    static thread_local pcl::EuclideanClusterComparator<PointType, pcl::Normal, pcl::Label> euclidean_cluster_comparator;
    pcl::EuclideanClusterComparator<PointType, pcl::Normal, pcl::Label>::Ptr euclidean_cluster_comparator_ptr(&euclidean_cluster_comparator,
                                                                                                              emptyDestructorForStack<pcl::EuclideanClusterComparator<PointType, pcl::Normal, pcl::Label> >);
    euclidean_cluster_comparator.setInputCloud(input);
    euclidean_cluster_comparator.setLabels(plane_labels); // set labels from plane segmentation
    euclidean_cluster_comparator.setExcludeLabels(excluded_indices); // set them exclude
    euclidean_cluster_comparator.setDistanceThreshold(parameters::euclidean_distance_threshold, false); // set radius

    pcl::OrganizedConnectedComponentSegmentation<PointType, pcl::Label> euclidean_segmentation(euclidean_cluster_comparator_ptr);
    euclidean_segmentation.setInputCloud(input);
    euclidean_segmentation.segment(euclidean_labels, euclidean_label_indices);
}


static inline void colorSegmentation(const CloudType::ConstPtr input,
                                     const pcl::PointIndices &roi_indices,
                                     std::vector<pcl::PointIndices> &color_segment_indices)
{


    pcl::search::KdTree<PointType> kd_tree;
    pcl::search::Search<PointType>::Ptr search_tree(&kd_tree, emptyDestructorForStack<pcl::search::Search<PointType> >);
    pcl::PointIndices::ConstPtr roi_indices_ptr(&roi_indices, emptyDestructorForStack<pcl::PointIndices>);

    pcl::RegionGrowingRGB<PointType> reg_grow;

    reg_grow.setInputCloud(input);
    reg_grow.setIndices(roi_indices_ptr);
    reg_grow.setSearchMethod(search_tree);
    reg_grow.setDistanceThreshold(parameters::region_growing_distance_threshold);
    reg_grow.setPointColorThreshold(parameters::region_growing_point_color_thresh);
    reg_grow.setRegionColorThreshold(parameters::region_growing_region_color_thresh);
    reg_grow.setMinClusterSize(parameters::region_growing_min_inliers);

    reg_grow.extract(color_segment_indices);
}


static inline void colorizeCloud(CloudType &cloud)
{
    uint8_t red   = rand() % 256,
            green = rand() % 256,
            blue  = rand() % 256;
    size_t cloud_sz = cloud.size();
    for(size_t i = 0; i < cloud_sz; ++i) {
        PointType &pt = cloud.points[i];
        pt.r = red;
        pt.g = green;
        pt.b = blue;
    }
}

static inline bool transformPointCloud(const CloudType &input, CloudType &output)
{
    static tf::TransformListener tf_listener;
    return pcl_ros::transformPointCloud(parameters::world_tf_frame, input, output, tf_listener);
}

#ifdef ENABLE_LOCATION_POSTFILTERING

namespace parameters {
Eigen::Vector4f post_filtering_region_min_pt; // Filtering region starting point
Eigen::Vector4f post_filtering_region_max_pt; // Filtering region ending point
double postfilter_single_axis_allowance; // How much excess space in a axis counted "in" region
double postfilter_volume_intersection_threshold; // Minimum volumetric intersection with region
}

static inline bool checkRangesOverlap(double range1_start, double range1_end,
                                      double range2_start, double range2_end)
{
    return (range2_start <= range1_end + parameters::postfilter_single_axis_allowance) &&
           (range1_start <= range2_end + parameters::postfilter_single_axis_allowance);
}

static inline double oneDIntersectionRate(double range1_start, double range1_end,
                                           double range2_start, double range2_end)
{
    double intersection = std::min(range1_end, range2_end) - std::max(range1_start, range2_start);
    double smaller_range   = std::min(range1_end - range1_start, range2_end - range2_start);
    return (intersection / smaller_range);
}

static inline double objectRegionIntersectionRate(Eigen::Vector4f segment_min, Eigen::Vector4f segment_max)
{
    using namespace parameters;
    bool intersectX = checkRangesOverlap(post_filtering_region_min_pt[0], post_filtering_region_max_pt[0], segment_min[0], segment_max[0]);
    double rateX = oneDIntersectionRate(post_filtering_region_min_pt[0], post_filtering_region_max_pt[0], segment_min[0], segment_max[0]);

    bool intersectY = checkRangesOverlap(post_filtering_region_min_pt[1], post_filtering_region_max_pt[1], segment_min[1], segment_max[1]);
    double rateY = oneDIntersectionRate(post_filtering_region_min_pt[1], post_filtering_region_max_pt[1], segment_min[1], segment_max[1]);

    bool intersectZ = checkRangesOverlap(post_filtering_region_min_pt[2], post_filtering_region_max_pt[2], segment_min[2], segment_max[2]);
    double rateZ = oneDIntersectionRate(post_filtering_region_min_pt[2], post_filtering_region_max_pt[2], segment_min[2], segment_max[2]);

    return (intersectX ? rateX : 0.0) *
           (intersectY ? rateY : 0.0) *
           (intersectZ ? rateZ : 0.0);
}
#endif

static inline void publishSegments(const CloudType::ConstPtr cloud, const std::vector<pcl::PointIndices> extracted_segment_indices)
{
    violet_msgs::DetectionInfo all_detections;
    CloudType vis_point_cloud;
    pcl::ExtractIndices<PointType> extractor;
    size_t num_total_segments = extracted_segment_indices.size();

    if(cloud->empty()) {
        ROS_WARN("WTF: Cloud is empty! Skipping");
        return;
    }

    // Run for all detected segments
    for (size_t i_segm = 0; i_segm < num_total_segments; ++i_segm) {
        CloudType current_segment, transformed_segment;
        pcl::PointIndices::ConstPtr indice_ptr(&(extracted_segment_indices[i_segm]), emptyDestructorForStack<pcl::PointIndices>);
        extractor.setInputCloud(cloud);
        extractor.setIndices(indice_ptr);
        extractor.setKeepOrganized(true);
        extractor.filter(current_segment);

        if(current_segment.empty()) {
            ROS_WARN("Current segment is empty! Skipping");
            continue;
        }

        struct { long long r, g, b; double norm_r, norm_g, norm_b; long long h, s, v; } average_color;
        average_color.r = 0;
        average_color.g = 0;
        average_color.b = 0;

        long long non_nan_points = 0;

        for(CloudType::const_iterator pt_it = current_segment.begin(), pt_end = current_segment.end(); pt_it != pt_end; ++pt_it ) {
            const PointType& cur_pt = *pt_it;
            //is nan aka not equal itself
            if(cur_pt.x != cur_pt.x || cur_pt.y != cur_pt.y || cur_pt.z != cur_pt.z) {
                continue;
            }
            average_color.r += cur_pt.r;
            average_color.g += cur_pt.g;
            average_color.b += cur_pt.b;
            non_nan_points++;
        }

        if(non_nan_points <= 0) {
            continue;
        }

        average_color.r /= non_nan_points;
        average_color.g /= non_nan_points;
        average_color.b /= non_nan_points;

        average_color.norm_r = static_cast<double>(average_color.r) / 255.0;
        average_color.norm_g = static_cast<double>(average_color.g) / 255.0;
        average_color.norm_b = static_cast<double>(average_color.b) / 255.0;

        long long rgb_max = std::max(average_color.r, std::max(average_color.g, average_color.b));
        long long rgb_min = std::min(average_color.r, std::min(average_color.g, average_color.b));
        long long rgb_delta = rgb_max - rgb_min;
        double norm_delta = static_cast<double>(rgb_delta) / 255.0;

        if(rgb_delta > 0) {
            double norm_h;
            if(rgb_max == average_color.r) {
                norm_h = std::fmod((average_color.norm_g - average_color.norm_b) / norm_delta, 6);
                average_color.s = rgb_delta / average_color.norm_r;
            }
            else if(rgb_max == average_color.b) {
                norm_h = 2 + ((average_color.norm_b - average_color.norm_r) / norm_delta);
                average_color.s = rgb_delta / average_color.norm_g;
            }
            else {
                norm_h = 4 + ((average_color.norm_r - average_color.norm_g) / norm_delta);
                average_color.s = rgb_delta / average_color.norm_b;
            }
            average_color.h = norm_h * 60.0;
        }
        else {
            //Colorless
            average_color.h = 0;
            average_color.s = 0;
        }
        average_color.v = rgb_max;

        std::string color_name;
        if(average_color.s > 80) {
            if(average_color.h < 30) {
                color_name = "red";
            }
            else if(average_color.h < 75) {
                color_name = "yellow";
            }
            else if(average_color.h < 200) {
                color_name = "green";
            }
            else if(average_color.h < 270) {
                color_name = "blue";
            }
            else {
                color_name = "red";
            }
        }
        else {
            if(average_color.v < 30) {
                color_name = "black";
            }
            else if(average_color.v < 200) {
                color_name = "grey";
            }
            else {
                color_name = "white";
            }
        }

        transformPointCloud(current_segment, transformed_segment);
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(transformed_segment, min_pt, max_pt);

#ifdef ENABLE_LOCATION_POSTFILTERING
        double intersection_rate = objectRegionIntersectionRate(min_pt, max_pt);
        if(intersection_rate < parameters::postfilter_volume_intersection_threshold ) {
            continue;
        }
#endif
        if(registered_to_violet) {
            violet_msgs::ObjectInfo violet_object;
            violet_msgs::ObjectProperty size_prop, location_prop,
                                        min_prop, max_prop, sensor_frame,
                                        color_values_prop, color_values_hsv_prop, color_name_prop;

            location_prop.attribute = "location";
            location_prop.data.resize(3);
            location_prop.data[0] = max_pt[0] - (max_pt[0] - min_pt[0])/2; // x
            location_prop.data[1] = max_pt[1] - (max_pt[1] - min_pt[1])/2; // y
            location_prop.data[2] = max_pt[2] - (max_pt[2] - min_pt[2])/2; // z
            violet_object.properties.push_back(location_prop);

            size_prop.attribute = "size";
            size_prop.data.push_back(max_pt[0] - min_pt[0]); // x
            size_prop.data.push_back(max_pt[1] - min_pt[1]); // y
            size_prop.data.push_back(max_pt[2] - min_pt[2]); // z
            violet_object.properties.push_back(size_prop);

            min_prop.attribute = "min";
            min_prop.data.push_back(min_pt[0]); // x
            min_prop.data.push_back(min_pt[1]); // y
            min_prop.data.push_back(min_pt[2]); // z
            violet_object.properties.push_back(min_prop);

            max_prop.attribute = "max";
            max_prop.data.push_back(max_pt[0]); // x
            max_prop.data.push_back(max_pt[1]); // y
            max_prop.data.push_back(max_pt[2]); // z
            violet_object.properties.push_back(max_prop);

            color_values_prop.attribute = "color_values";
            color_values_prop.data.push_back(average_color.r);
            color_values_prop.data.push_back(average_color.g);
            color_values_prop.data.push_back(average_color.b);
            violet_object.properties.push_back(color_values_prop);

            color_values_hsv_prop.attribute = "color_values_hsv";
            color_values_hsv_prop.data.push_back(average_color.h);
            color_values_hsv_prop.data.push_back(average_color.s);
            color_values_hsv_prop.data.push_back(average_color.v);
            violet_object.properties.push_back(color_values_hsv_prop);

            sensor_frame.attribute = "sensor_frame";
            sensor_frame.values.push_back(parameters::camera_tf_frame);
            violet_object.properties.push_back(sensor_frame);

            color_name_prop.attribute = "color_name";
            color_name_prop.values.push_back(color_name);
            violet_object.properties.push_back(color_name_prop);

            all_detections.objects.push_back(violet_object);
        }

        //colorizeCloud(current_segment);
        vis_point_cloud += current_segment;
    }

    sensor_msgs::PointCloud2 vis_cloud_ros;
    pcl::toROSMsg(vis_point_cloud, vis_cloud_ros);
    vis_cloud_ros.header.frame_id = parameters::camera_tf_frame;
    vis_cloud_pub.publish(vis_cloud_ros);

    if(registered_to_violet) {
        all_detections.header.frame_id = parameters::world_tf_frame;
        all_detections.header.stamp = ros::Time::now();
        segmented_violet_obj_pub.publish(all_detections);
    }

}


void pointCloudCallback(const CloudType::ConstPtr &msg_cloud)
{
   processPointCloud(msg_cloud);
}

void dynamicConfigCallback(airlab_segment::SegmentParamsConfig &config, uint32_t)
{
    using namespace parameters;
    min_depth = config.groups.passthrough_prefiltering.passthrough_min_depth;
    max_depth = config.groups.passthrough_prefiltering.passthrough_max_depth;

    plane_min_inliers   = config.groups.multiplane_segmentation.plane_min_inliers;

    euclidean_distance_threshold = config.groups.euclidean_clustering.cluster_distance_threshold;
    cluster_min_inliers = config.groups.euclidean_clustering.cluster_min_inliers;
    cluster_max_inliers = config.groups.euclidean_clustering.cluster_max_inliers;

    region_growing_min_inliers         = config.groups.region_growing_segmentation.region_growing_min_inliers;
    region_growing_point_color_thresh  = config.groups.region_growing_segmentation.region_growing_point_color_threshold;
    region_growing_region_color_thresh = config.groups.region_growing_segmentation.region_growing_region_color_threshold;
    region_growing_distance_threshold  = config.groups.region_growing_segmentation.region_growing_distance_threshold;

#ifdef ENABLE_LOCATION_POSTFILTERING
    post_filtering_region_min_pt = Eigen::Vector4f(config.groups.segment_location_postfiltering.post_filtering_min_point_x,
                                                   config.groups.segment_location_postfiltering.post_filtering_min_point_y,
                                                   config.groups.segment_location_postfiltering.post_filtering_min_point_z,
                                                   0.0);
    post_filtering_region_max_pt = Eigen::Vector4f(config.groups.segment_location_postfiltering.post_filtering_max_point_x,
                                                   config.groups.segment_location_postfiltering.post_filtering_max_point_y,
                                                   config.groups.segment_location_postfiltering.post_filtering_max_point_z,
                                                   0.0);
    postfilter_volume_intersection_threshold = config.groups.segment_location_postfiltering.post_filtering_volume_intersection_threshold;
    postfilter_single_axis_allowance = config.groups.segment_location_postfiltering.post_filtering_single_axis_intersection_allowance;
#endif
}


}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "pcl_euclidian_segmenter");
    ros::NodeHandle nh;
    ros::NodeHandle loc_nh("~");

    /* Read Params */
    parameters::input_pc_topic = loc_nh.param<std::string>("input_cloud_topic", "camera/depth_registered/points");
    parameters::camera_tf_frame = loc_nh.param<std::string>("camera_tf_frame", "camera_rgb_optical_frame");
    parameters::world_tf_frame = loc_nh.param<std::string>("world_tf_frame", "world");
    parameters::thread_count = loc_nh.param<int>("thread_count", 4);
    parameters::queue_size = loc_nh.param<int>("queue_size", 8);
    bool register_violet = loc_nh.param<bool>("register_to_violet", true);
    std::string violet_ns = loc_nh.param<std::string>("violet_namespace", "violet");

    /* Initialize Dynamic reconfigure Server */
    dynamic_reconfigure::Server<airlab_segment::SegmentParamsConfig> server;
    server.setCallback(dynamicConfigCallback);


    if(register_violet) {
        parameters::violet_publish_topic = loc_nh.param<std::string>("violet_detections_topic", "segment_detections");
        ROS_INFO("Waiting for Violet starts up");
        bool service_exists = false;
        for ( int i = 0; i < 3 && !(service_exists = ros::service::exists(violet_ns + "/register_source", false)); ++i) { boost::this_thread::sleep_for(boost::chrono::milliseconds(200)); }

        if(service_exists) {
            ros::ServiceClient registrationClient = nh.serviceClient<violet_srvs::RegisterSource>(violet_ns + "/register_source");
            violet_srvs::RegisterSource register_srv;

            register_srv.request.topic_name = parameters::violet_publish_topic;
            register_srv.request.source_algorithm_name = "Segmentation";
            registered_to_violet = registrationClient.call(register_srv);
            ROS_INFO("Violet registration is %s ", (registered_to_violet ? "successful" : "unsucessful") );
        }
    }

    parameters::vis_pc_publish_topic = loc_nh.param<std::string>("visualization_cloud_topic", "segmented_cloud");
    vis_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(parameters::vis_pc_publish_topic, 10);
    if(registered_to_violet) {
        segmented_violet_obj_pub = nh.advertise<violet_msgs::DetectionInfo>(parameters::violet_publish_topic, 10);
    }
    ros::Subscriber sub = nh.subscribe(parameters::input_pc_topic, parameters::queue_size, pointCloudCallback);

    ros::MultiThreadedSpinner spinner(parameters::thread_count);
    spinner.spin();
    return 0;
}
