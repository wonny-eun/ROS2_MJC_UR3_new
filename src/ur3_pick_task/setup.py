import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ur3_pick_task'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config', 'ik'), glob('config/ik/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wonny',
    maintainer_email='wonny@todo.todo',
    description='UR3 MuJoCo pick task with OpenCV and MoveIt MoveGroup action',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ur3_pick_task = ur3_pick_task.pick_task_script:main',
            'tcp_ik_solver = ur3_pick_task.tcp_ik_solver:main',
        ],
    },
)
