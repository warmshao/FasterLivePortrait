# -*- coding: utf-8 -*-
# @Time    : 2024/7/13 11:21
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : RealTimeLivePortrait
# @FileName: run.py

import cv2
import os

"""
:wq
pip install onnxruntime-gpu -U --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple

./build.sh --parallel --build_shared_lib --use_cuda --cuda_version 12.1 --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/ --config Release --build_wheel --skip_tests --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc --allow_running_as_root
"""

if __name__ == '__main__':
    pass
