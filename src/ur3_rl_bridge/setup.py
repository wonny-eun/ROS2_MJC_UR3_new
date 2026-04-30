import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ur3_rl_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config', 'handeye'), glob('config/handeye/*.yaml')),
        (os.path.join('share', package_name, 'config', 'tcp'), glob('config/tcp/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wonny',
    maintainer_email='wonny@todo.todo',
    description='ROS2 bridge utilities for UR3 MuJoCo sim2real pipeline',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'virtual_perception_node = ur3_rl_bridge.virtual_perception_node:main',
            'mujoco_object_manager_node = ur3_rl_bridge.mujoco_object_manager_node:main',
            'mujoco_rl_camera_preview = ur3_rl_bridge.mujoco_rl_camera_preview_node:main',
            'multi_view_cloud_fusion_node = ur3_rl_bridge.multi_view_cloud_fusion_node:main',
            'yolo_object_preview = ur3_rl_bridge.yolo_object_preview_node:main',
            'noisy_camera_node = ur3_rl_bridge.noisy_camera_node:main',
            'handeye_tf_publisher = ur3_rl_bridge.handeye_tf_publisher_node:main',
            'tcp_task_pose_test = ur3_rl_bridge.tcp_task_pose_test_node:main',
        ],
    },
)
