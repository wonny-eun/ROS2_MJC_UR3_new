import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. 경로 설정
    ur3_urdf_path = '/home/wonny/ur3_control/install/ur_description/share/ur_description/urdf/ur3.urdf'
    
    # 2. mujoco_ros2_control_demos의 demo.launch.py 가져오기
    demo_launch_dir = os.path.join(get_package_share_directory('mujoco_ros2_control_demos'), 'launch')
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(demo_launch_dir, 'demo.launch.py')),
            launch_arguments={
                'mujoco_model_file': ur3_urdf_path,
                'robot_name': 'ur3'
            }.items(),
        )
    ])

