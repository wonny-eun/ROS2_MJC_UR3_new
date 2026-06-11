// SPDX-License-Identifier: MIT
// Relay sensor_msgs/CameraInfo -> negotiated NitrosCameraInfo for Isaac FoundationPose.

#include <memory>
#include <string>

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace ur3_isaac_nitros_bridge
{

class RosCameraInfoNitrosRelay : public rclcpp::Node
{
public:
  explicit RosCameraInfoNitrosRelay(const rclcpp::NodeOptions & options)
  : Node("ros_camera_info_nitros_relay", options)
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "/rl_camera/camera_info");
    output_topic_ = declare_parameter<std::string>("output_topic", "/rgb/camera_info");

    nitros_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosCameraInfo>>(
      this, output_topic_, "nitros_camera_info", {}, rclcpp::SensorDataQoS());

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RosCameraInfoNitrosRelay::onCameraInfo, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "ROS->Nitros camera_info relay %s -> %s",
      input_topic_.c_str(), output_topic_.c_str());
  }

private:
  void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
  {
    nvidia::isaac_ros::nitros::NitrosCameraInfo nitros_info;
    rclcpp::TypeAdapter<
      nvidia::isaac_ros::nitros::NitrosCameraInfo,
      sensor_msgs::msg::CameraInfo>::convert_to_custom(*msg, nitros_info);
    nitros_pub_->publish(nitros_info);
  }

  std::string input_topic_;
  std::string output_topic_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosCameraInfo>>
  nitros_pub_;
};

}  // namespace ur3_isaac_nitros_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(ur3_isaac_nitros_bridge::RosCameraInfoNitrosRelay)
