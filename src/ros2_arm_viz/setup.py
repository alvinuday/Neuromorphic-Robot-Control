from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ros2_arm_viz'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='ROS2 visualization for 2-DOF planar arm with MPC',
    license='MIT',
    entry_points={
        'console_scripts': [
            'joint_state_mpc_node = ros2_arm_viz.joint_state_mpc_node:main',
            'robot_description_publisher_node = ros2_arm_viz.robot_description_publisher_node:main',
        ],
    },
)
