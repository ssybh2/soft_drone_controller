<div align="center">

# Soft Drone Controller

**ROS 2 control stack for a soft ducted quadrotor**

<p>
  <img src="https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white" alt="ROS 2 Humble">
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white" alt="Python 3">
  <img src="https://img.shields.io/badge/Control-Quaternion%20PID-6B7280" alt="Quaternion PID">
  <img src="https://img.shields.io/badge/Interface-EtherCAT%20%2B%20DShot-2563EB" alt="EtherCAT and DShot">
</p>

<img src="docs/assets/soft_drone_hero.jpg" width="820" alt="Soft ducted quadrotor research platform">

</div>

## Overview

`soft_drone_controller` is a ROS 2 controller for an experimental soft ducted quadrotor. The stack combines **quaternion attitude PID**, **position hold / path control**, and **EtherCAT + DShot** motor output for real-flight experiments.

Main nodes include `drone_controller`, `position_control`, `position_cmd`, and `send_target`.

## Demo

<div align="center">
  <img src="docs/assets/demo-placeholder.svg" width="760" alt="Flight demo placeholder">
  <br>
  <sub>Demo video coming soon · upload docs/assets/demo.mov or demo.mp4</sub>
</div>

## Quick Start

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch soft_drone_controller controller_launch.py
```

---

<div align="center">
  <sub>Soft Robotics · Aerial Robotics · ROS 2 · EtherCAT</sub>
</div>
