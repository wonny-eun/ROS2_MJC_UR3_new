#!/bin/sh
# CMake FindCUDA only needs a working --version during configure (no full toolkit required).
if [ "$1" = "--version" ]; then
  echo "nvcc: NVIDIA (fake) release 12.6, V12.6.0"
  exit 0
fi
exit 0
