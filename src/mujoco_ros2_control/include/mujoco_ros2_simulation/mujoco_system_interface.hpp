/**
 * Copyright (c) 2025, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 *
 * All rights reserved.
 *
 * This software is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <mujoco/mujoco.h>

// Pull in the Simulate class and PhysicsThread/RenderLoop declarations:
#include "glfw_adapter.h"  // for mj::GlfwAdapter
#include "simulate.h"      // must be on your include path, handled by CMake

#include "mujoco_ros2_simulation/data.hpp"
#include "mujoco_ros2_simulation/mujoco_cameras.hpp"
#include "mujoco_ros2_simulation/mujoco_lidar.hpp"

namespace mujoco_ros2_simulation
{
class MujocoSystemInterface : public hardware_interface::SystemInterface
{
public:
  /**
   * @brief ros2_control SystemInterface to wrap Mujocos Simulate application.
   *
   * Supports Actuators, Force Torque/IMU Sensors, and RGB-D camera, and Lidar Sensors in ROS 2 simulations.
   * For more information on configuration refer to the docs, check the comment strings below, and refer to
   * the example in the test folder.
   */
  MujocoSystemInterface();
  ~MujocoSystemInterface() override;

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

  hardware_interface::return_type perform_command_mode_switch(const std::vector<std::string>& start_interfaces,
                                                              const std::vector<std::string>& stop_interfaces) override;

  hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
  hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  /**
   * @brief Returns a copy of the MuJoCo model.
   *
   * This method locks the simulation mutex to ensure thread safety.
   * @param dest Pointer to an mjModel structure where the copy will be stored. The pointer will be allocated if it is nullptr.
   */
  void get_model(mjModel*& dest);

  /**
   * @brief Returns a copy of the current MuJoCo data.
   *
   * This method locks the simulation mutex to ensure thread safety.
   * @param dest Pointer to an mjData structure where the copy will be stored. The pointer will be allocated if it is nullptr.
   */
  void get_data(mjData*& dest);

  /**
   * @brief Sets the MuJoCo data to the provided value.
   *
   * This method locks the simulation mutex to ensure thread safety.
   * @param mj_data Pointer to an mjData structure containing the new data.
   */
  void set_data(mjData* mj_data);

private:
  /**
   * @brief Loads actuator information into the HW interface.
   *
   * Will pull joint/actuator information from the provided HardwareInfo, and map it to the appropriate
   * actuator in the sim's mujoco data. The data wrappers will be used as control/state interfaces for
   * the HW interface.
   */
  void register_joints(const hardware_interface::HardwareInfo& info);

  /**
   * @brief Constructs all sensor data containers for the interface
   *
   * Pulls sensors (FTS and IMUs) out of the HardwareInfo and uses it to map relevant data containers
   * in the ros2_control interface. There are expectations on the naming of sensors in both the MJCF and
   * the ros2_control xacro, as Mujoco does not have direct support for either of these sensors.
   *
   * For a FTS named <FTS>, we add both a `force` and `torque` sensor to the MJCF as:
   *
   *  <sensor>
   *    <force name="<FTS>_force" site="ft_frame"/>
   *    <torque name="<FTS>_torque" site="ft_frame"/>
   *  </sensor>
   *
   * In the ROS 2 control xacro, these must be mapped to a state interface called `<FTS>_fts`, so,
   *
   *  <sensor name="<FTS>_fts">
   *    <state_interface name="force.x"/>
   *    <state_interface name="force.y"/>
   *    <state_interface name="force.z"/>
   *    <state_interface name="torque.x"/>
   *    <state_interface name="torque.y"/>
   *    <state_interface name="torque.z"/>
   *    <param name="frame_id">fts_sensor</param>
   *  </sensor>
   *
   * The HW interface will map the state interfaces accordingly. Similarly for an IMU named <IMU>, we
   * must add three separate sensors to the MJCF, update the site/obj accordingly:
   *
   *  <sensor>
   *    <framequat name="<IMU>_quat" objtype="site" objname="obj_imu" />
   *    <gyro name="<IMU>_gyro" site="obj_imu" />
   *    <accelerometer name="<IMU>_accel" site="obj_imu" />
   *  </sensor>
   *
   * These can be mapped with the following xacro (note the `_imu` suffix):
   *
   *  <sensor name="<IMU>_imu">
   *    <state_interface name="orientation.x"/>
   *    <state_interface name="orientation.y"/>
   *    <state_interface name="orientation.z"/>
   *    <state_interface name="orientation.w"/>
   *    <state_interface name="angular_velocity.x"/>
   *    <state_interface name="angular_velocity.y"/>
   *    <state_interface name="angular_velocity.z"/>
   *    <state_interface name="linear_acceleration.x"/>
   *    <state_interface name="linear_acceleration.y"/>
   *    <state_interface name="linear_acceleration.z"/>
   *  </sensor>
   */
  void register_sensors(const hardware_interface::HardwareInfo& info);

  /**
   * @brief Sets the initial simulation conditions (pos, vel, ctrl) values from provided filepath.
   *
   * @param override_start_position_file filepath that contains starting positions
   * @return success of reading the file and setting the positions
   */
  bool set_override_start_positions(const std::string& override_start_position_file);

  /**
   * @brief Set the initial pose for all actuators if provided in the URDF.
   */
  void set_initial_pose();

  /**
   * @brief Spins the physics simulation for the Simulate Application
   */
  void PhysicsLoop();

  /**
   * @brief Publishes the Simulate Application's timestamp to the /clock topic
   *
   * This enables pausing and restarting of the simulation through the application window.
   */
  void publish_clock();
  void publish_object_transforms();
  void publish_object_markers();

  void register_module_plate_weld_interfaces();
  void on_module_plate_weld_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void attach_module1_plate_to_case_base_srv(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void reset_module1_plate_table_hold_srv(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  // System information
  hardware_interface::HardwareInfo system_info_;
  std::string model_path_;

  // MuJoCo data pointers
  mjModel* mj_model_{ nullptr };
  mjData* mj_data_{ nullptr };

  // For rendering
  mjvCamera cam_;
  mjvOption opt_;
  mjvPerturb pert_;

  // Speed scaling parameter. if set to >0 then we ignore the value set in the simulate app and instead
  // attempt to loop at whatever this is set to. If this is <0, then we use the value from the app.
  double sim_speed_factor_;

  // Primary simulate object
  std::unique_ptr<mujoco::Simulate> sim_;

  // Threads for rendering physics, the UI simulation, and the ROS node
  std::thread physics_thread_;
  std::thread ui_thread_;

  // Provides access to ROS interfaces for elements that require it
  std::shared_ptr<rclcpp::Node> mujoco_node_;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread executor_thread_;

  // Primary clock publisher for the world
  std::shared_ptr<rclcpp::Publisher<rosgraph_msgs::msg::Clock>> clock_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>> object_marker_publisher_;
  std::vector<std::pair<std::string, std::string>> object_tf_name_pairs_;
  std::unordered_map<std::string, int> object_body_ids_;
  std::unordered_map<std::string, int> object_geom_ids_;
  std::unordered_map<std::string, std::string> object_mesh_resource_;
  double object_tf_publish_period_s_{ 1.0 / 60.0 };
  double object_tf_last_pub_sim_time_{ -1.0 };

  // Containers for RGB-D cameras
  std::unique_ptr<MujocoCameras> cameras_;

  // Containers for LIDAR sensors
  std::unique_ptr<MujocoLidar> lidar_sensors_;

  // Optional: weld Module_1_Plate to Case_Base (MJCF equality + ROS command)
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr module_plate_weld_pose_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr module_plate_attach_case_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr module_plate_reset_table_srv_;
  std::mutex module_plate_weld_cmd_mu_;
  geometry_msgs::msg::PoseStamped module_plate_weld_pose_cmd_;
  bool module_plate_weld_have_cmd_{ false };
  bool module_plate_weld_registered_{ false };
  int module_plate_weld_world_eq_{ -1 };
  int module_plate_weld_case_eq_{ -1 };

  // Mutex used inside simulate.h for protecting model/data, we keep a reference
  // here to protect access to shared data.
  // TODO: It would be far better to put all relevant data into a single container with accessors
  //       in a common location rather than passing around the raw pointer to the mutex, but it would
  //       require more work to pull it out of simulate.h.
  std::recursive_mutex* sim_mutex_{ nullptr };

  // Data containers for the HW interface
  std::unordered_map<std::string, hardware_interface::ComponentInfo> joint_hw_info_;
  std::unordered_map<std::string, hardware_interface::ComponentInfo> sensors_hw_info_;

  // Data containers for the HW interface
  std::vector<JointState> joint_states_;
  std::vector<FTSensorData> ft_sensor_data_;
  std::vector<IMUSensorData> imu_sensor_data_;
};

}  // namespace mujoco_ros2_simulation
