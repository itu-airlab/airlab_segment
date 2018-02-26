#include <string>
#include <unordered_map>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <violet_msgs/DetectionInfo.h>
#include <visualization_msgs/MarkerArray.h>

struct CallbackFunctor
{
    struct Object
    {
        tf::Point center;
        tf::Point size;
        tf::Point max_point;
        tf::Point min_point;
        struct {int r; int g; int b;} color;
    };

    static std::unordered_map<std::string, tf::Point> cached_object_sizes;
    static tf::TransformListener *tf_listener;
    static std::string world_frame;

    std::string source;
    std::vector<Object> objects;
    ros::Publisher publisher;
    void callback(const violet_msgs::DetectionInfo::ConstPtr &msg);
    void publishMarkers();
};


std::unordered_map<std::string, tf::Point> CallbackFunctor::cached_object_sizes;
tf::TransformListener *CallbackFunctor::tf_listener = nullptr;
std::string CallbackFunctor::world_frame;

void CallbackFunctor::callback(const violet_msgs::DetectionInfo::ConstPtr &msg)
{
    if(!tf_listener->waitForTransform(world_frame, msg->header.frame_id, ros::Time(0), ros::Duration(2))) {
        return;
    }
    objects.clear();
    tf::StampedTransform transform;
    tf_listener->lookupTransform(world_frame, msg->header.frame_id, ros::Time(0), transform);
    Object to_add;
    for(auto &current_obj : msg->objects) {
        tf::Point untransformed_loc;
        for(auto &current_prop : current_obj.properties) {
            if(current_prop.attribute == "location") {
                untransformed_loc = tf::Point(current_prop.data[0], current_prop.data[1], current_prop.data[2]);
                to_add.center = untransformed_loc; //transform * untransformed_loc ;
            }
            else if(current_prop.attribute == "size") {
                tf::Point untransformed_size(current_prop.data[0], current_prop.data[1], current_prop.data[2]);
                tf::Point start_pt(transform * (untransformed_loc - untransformed_size/2));
                tf::Point end_pt(transform * (untransformed_loc + untransformed_size/2));
                to_add.size = untransformed_size; //end_pt - start_pt ;
            }
            else if (current_prop.attribute == "max") {
                tf::Point max_pt(current_prop.data[0], current_prop.data[1], current_prop.data[2]);
                to_add.max_point = max_pt;
            }
            else if (current_prop.attribute == "min") {
                tf::Point min_pt(current_prop.data[0], current_prop.data[1], current_prop.data[2]);
                to_add.min_point = min_pt;
            }
            else if (current_prop.attribute == "color_values") {
                to_add.color.r = current_prop.data[0];
                to_add.color.g = current_prop.data[1];
                to_add.color.b = current_prop.data[2];
            }
        }
        objects.push_back(to_add);
    }
}

void CallbackFunctor::publishMarkers()
{
    visualization_msgs::MarkerArray marr;
    visualization_msgs::Marker deleteall_marker;
    deleteall_marker.action = 3; //DELETEALL Although it says it is supported in indigo it's not defined!
    marr.markers.push_back(deleteall_marker);
    for(int i = 0; i < objects.size(); ++i) {
        Object& obj = objects[i];

        visualization_msgs::Marker bounding_box;
        visualization_msgs::Marker centeroid_sphere;
        visualization_msgs::Marker max_sphere;
        visualization_msgs::Marker min_sphere;
        visualization_msgs::Marker name_text;

        bounding_box.action = visualization_msgs::Marker::ADD;
        bounding_box.header.frame_id = world_frame;
        bounding_box.type = visualization_msgs::Marker::CUBE;
        bounding_box.ns = "bounding_boxes";
        bounding_box.id = i;

        bounding_box.scale.x = obj.size.x();
        bounding_box.scale.y = obj.size.y();
        bounding_box.scale.z = obj.size.z();

        bounding_box.pose.position.x = obj.center.x();
        bounding_box.pose.position.y = obj.center.y();
        bounding_box.pose.position.z = obj.center.z();

        bounding_box.pose.orientation.x = 0.0;
        bounding_box.pose.orientation.y = 0.0;
        bounding_box.pose.orientation.z = 0.0;
        bounding_box.pose.orientation.w = 1.0;

        bounding_box.color.r = double(obj.color.r)/255;
        bounding_box.color.g = double(obj.color.g)/255;
        bounding_box.color.b = double(obj.color.b)/255;
        bounding_box.color.a = 0.5;

        marr.markers.push_back(bounding_box);

        centeroid_sphere.action = visualization_msgs::Marker::ADD;
        centeroid_sphere.header.frame_id = world_frame;
        centeroid_sphere.type = visualization_msgs::Marker::SPHERE;
        centeroid_sphere.ns = "centeroid_spheres";
        centeroid_sphere.id = i;

        centeroid_sphere.scale.x = 0.01;
        centeroid_sphere.scale.y = 0.01;
        centeroid_sphere.scale.z = 0.01;

        centeroid_sphere.pose.position.x = obj.center.x();
        centeroid_sphere.pose.position.y = obj.center.y();
        centeroid_sphere.pose.position.z = obj.center.z();

        centeroid_sphere.pose.orientation.x = 0.0;
        centeroid_sphere.pose.orientation.y = 0.0;
        centeroid_sphere.pose.orientation.z = 0.0;
        centeroid_sphere.pose.orientation.w = 1.0;

        centeroid_sphere.color.r = 1.0;
        centeroid_sphere.color.g = 0.0;
        centeroid_sphere.color.b = 0.0;
        centeroid_sphere.color.a = 0.7;

        marr.markers.push_back(centeroid_sphere);


        max_sphere.action = visualization_msgs::Marker::ADD;
        max_sphere.header.frame_id = world_frame;
        max_sphere.type = visualization_msgs::Marker::SPHERE;
        max_sphere.ns = "max_spheres";
        max_sphere.id = i;

        max_sphere.scale.x = 0.01;
        max_sphere.scale.y = 0.01;
        max_sphere.scale.z = 0.01;

        max_sphere.pose.position.x = obj.max_point.x();
        max_sphere.pose.position.y = obj.max_point.y();
        max_sphere.pose.position.z = obj.max_point.z();

        max_sphere.pose.orientation.x = 0.0;
        max_sphere.pose.orientation.y = 0.0;
        max_sphere.pose.orientation.z = 0.0;
        max_sphere.pose.orientation.w = 1.0;

        max_sphere.color.r = 0.0;
        max_sphere.color.g = 1.0;
        max_sphere.color.b = 0.0;
        max_sphere.color.a = 0.7;

        marr.markers.push_back(max_sphere);

        min_sphere.action = visualization_msgs::Marker::ADD;
        min_sphere.header.frame_id = world_frame;
        min_sphere.type = visualization_msgs::Marker::SPHERE;
        min_sphere.ns = "min_spheres";
        min_sphere.id = i;

        min_sphere.scale.x = 0.01;
        min_sphere.scale.y = 0.01;
        min_sphere.scale.z = 0.01;

        min_sphere.pose.position.x = obj.min_point.x();
        min_sphere.pose.position.y = obj.min_point.y();
        min_sphere.pose.position.z = obj.min_point.z();

        min_sphere.pose.orientation.x = 0.0;
        min_sphere.pose.orientation.y = 0.0;
        min_sphere.pose.orientation.z = 0.0;
        min_sphere.pose.orientation.w = 1.0;

        min_sphere.color.r = 0.0;
        min_sphere.color.g = 0.0;
        min_sphere.color.b = 1.0;
        min_sphere.color.a = 0.7;

        marr.markers.push_back(min_sphere);

    }
    publisher.publish(marr);
}



int main(int argc, char *argv[])
{
    ros::init(argc, argv, "segment_marker_publisher");
    ROS_INFO("Segment marker publisher has started");

    ros::NodeHandle nh;

    std::string segment_topic = "/segment_detections";
    CallbackFunctor segment_cb;
    segment_cb.publisher = nh.advertise<visualization_msgs::MarkerArray>("segment_visualization/markers", 1);

    ros::Subscriber segment_detection_sub = nh.subscribe(segment_topic, 1, &CallbackFunctor::callback, &segment_cb);

    CallbackFunctor::tf_listener = new tf::TransformListener;

    ros::NodeHandle local_nh("~");
    CallbackFunctor::world_frame = local_nh.param<std::string>("world_frame", "base");

    ROS_INFO("World frame is %s", CallbackFunctor::world_frame.c_str());

    ros::Rate rate(10);

    while(ros::ok()) {
        ros::spinOnce();
        segment_cb.publishMarkers();
        rate.sleep();
    }

    delete CallbackFunctor::tf_listener;

    return 0;
}
