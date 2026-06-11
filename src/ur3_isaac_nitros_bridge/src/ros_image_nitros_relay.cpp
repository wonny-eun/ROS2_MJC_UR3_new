// SPDX-License-Identifier: MIT
// Relay sensor_msgs/Image -> negotiated NitrosImage (rgb8 or 32FC1) for Isaac FoundationPose.

#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace ur3_isaac_nitros_bridge
{

class RosImageNitrosRelay : public rclcpp::Node
{
public:
  explicit RosImageNitrosRelay(const rclcpp::NodeOptions & options)
  : Node("ros_image_nitros_relay", options)
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "/rl_camera/noisy/color");
    output_topic_ = declare_parameter<std::string>("output_topic", "/rgb/image_rect_color");
    nitros_format_ = declare_parameter<std::string>("nitros_format", "nitros_image_rgb8");
    convert_bgr_to_rgb_ = declare_parameter<bool>("convert_bgr_to_rgb", true);

    nitros_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
        nvidia::isaac_ros::nitros::NitrosImage>>(
      this, output_topic_, nitros_format_, {}, rclcpp::SensorDataQoS());

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&RosImageNitrosRelay::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "ROS->Nitros image relay %s -> %s (%s)",
      input_topic_.c_str(), output_topic_.c_str(), nitros_format_.c_str());
  }

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    sensor_msgs::msg::Image ros_image = *msg;
    if (convert_bgr_to_rgb_ && ros_image.encoding == sensor_msgs::image_encodings::BGR8) {
      ros_image.encoding = sensor_msgs::image_encodings::RGB8;
      const size_t n = ros_image.data.size();
      for (size_t i = 0; i + 2 < n; i += 3) {
        std::swap(ros_image.data[i], ros_image.data[i + 2]);
      }
    }

    nvidia::isaac_ros::nitros::NitrosImage nitros_image;
    rclcpp::TypeAdapter<
      nvidia::isaac_ros::nitros::NitrosImage,
      sensor_msgs::msg::Image>::convert_to_custom(ros_image, nitros_image);
    nitros_pub_->publish(nitros_image);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string nitros_format_;
  bool convert_bgr_to_rgb_{true};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>>
  nitros_pub_;
};

}  // namespace ur3_isaac_nitros_bridge

RCLCPP_COMPONENTS_REGISTER_NODE(ur3_isaac_nitros_bridge::RosImageNitrosRelay)
