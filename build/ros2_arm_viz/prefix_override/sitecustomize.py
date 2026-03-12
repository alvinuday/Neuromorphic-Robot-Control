import sys
if sys.prefix == '/Users/alvin/y/envs/ros_env':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/install/ros2_arm_viz"
