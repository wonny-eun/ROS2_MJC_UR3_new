#!/bin/bash
# MuJoCo simulate 소스를 가져오는 공식 로직
MUJOCO_VERSION="main"
mkdir -p src/simulate
wget -P src/simulate https://raw.githubusercontent.com/google-deepmind/mujoco/main/simulate/simulate.cc
wget -P src/simulate https://raw.githubusercontent.com/google-deepmind/mujoco/main/simulate/simulate.h
wget -P src/simulate https://raw.githubusercontent.com/google-deepmind/mujoco/main/simulate/glfw_adapter.cc
wget -P src/simulate https://raw.githubusercontent.com/google-deepmind/mujoco/main/simulate/glfw_adapter.h
echo "MuJoCo simulate files updated successfully."
