cmake_minimum_required(VERSION 2.8.3)
project(airlab_segment)

find_package(catkin REQUIRED COMPONENTS
  violet_msgs
  violet_srvs
  dynamic_reconfigure
  pcl_msgs
  pcl_conversions
  pcl_ros
)
find_package(Eigen3 REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/SegmentParams.cfg
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs dynamic_reconfigure violet_msgs violet_srvs pcl_conversions pcl_ros
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

add_executable(euclidean_segmenter src/pcl_euclidean_segmenter.cpp)
add_dependencies(euclidean_segmenter ${catkin_EXPORTED_TARGETS})
target_link_libraries(euclidean_segmenter ${catkin_LIBRARIES})


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# Mark executables and/or libraries for installation
install(TARGETS airlab_segment euclidean_segmenter
 ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
 LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
 RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

