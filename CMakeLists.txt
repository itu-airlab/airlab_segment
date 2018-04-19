cmake_minimum_required(VERSION 2.8.3)
project(airlab_segment)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  violet_msgs
  violet_srvs
  dynamic_reconfigure
  pcl_msgs
  pcl_ros
  tf
)

generate_dynamic_reconfigure_options(
  cfg/SegmentParams.cfg
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs dynamic_reconfigure violet_msgs violet_srvs pcl_ros tf
)

option(ENABLE_LOCATION_POSTFILTERING "Compile segment location postfiltering step" ON)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

if(ENABLE_LOCATION_POSTFILTERING)
    add_definitions(-DENABLE_LOCATION_POSTFILTERING)
endif()

add_executable(euclidean_segmenter src/pcl_euclidean_segmenter.cpp)
add_dependencies(euclidean_segmenter ${catkin_EXPORTED_TARGETS})
target_link_libraries(euclidean_segmenter ${catkin_LIBRARIES})

add_executable(segment_marker_publisher src/segment_marker_publisher.cpp)
set_property(SOURCE src/segment_marker_publisher.cpp APPEND_STRING PROPERTY COMPILE_FLAGS "-std=c++11")
add_dependencies(segment_marker_publisher ${catkin_EXPORTED_TARGETS})
target_link_libraries(segment_marker_publisher ${catkin_LIBRARIES})


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# Mark executables and/or libraries for installation
#install(TARGETS airlab_segment euclidean_segmenter
# ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

