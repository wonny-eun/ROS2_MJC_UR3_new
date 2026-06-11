cmake_minimum_required(VERSION 3.8)
project(ur3_isaac_nitros_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Isaac ROS cmake pulls FindCUDA; provide minimal stubs when nvcc is not installed.
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_CUDART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
set(CUDA_NVCC_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fake_nvcc.sh")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(isaac_ros_managed_nitros REQUIRED)
find_package(isaac_ros_nitros_image_type REQUIRED)
find_package(isaac_ros_nitros_camera_info_type REQUIRED)

add_library(ros_image_nitros_relay SHARED src/ros_image_nitros_relay.cpp)
ament_target_dependencies(ros_image_nitros_relay
  rclcpp rclcpp_components sensor_msgs
  isaac_ros_managed_nitros isaac_ros_nitros_image_type)
rclcpp_components_register_nodes(ros_image_nitros_relay
  "ur3_isaac_nitros_bridge::RosImageNitrosRelay")

add_library(ros_camera_info_nitros_relay SHARED src/ros_camera_info_nitros_relay.cpp)
ament_target_dependencies(ros_camera_info_nitros_relay
  rclcpp rclcpp_components sensor_msgs
  isaac_ros_managed_nitros isaac_ros_nitros_camera_info_type)
rclcpp_components_register_nodes(ros_camera_info_nitros_relay
  "ur3_isaac_nitros_bridge::RosCameraInfoNitrosRelay")

install(TARGETS ros_image_nitros_relay ros_camera_info_nitros_relay
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_package()
