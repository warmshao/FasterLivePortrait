#!/bin/bash

# warping+spade model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/warping_spade-fix-v1.1.onnx
# motion_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/motion_extractor-v1.1.onnx -p fp32
# appearance_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/appearance_feature_extractor-v1.1.onnx
# stitching model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching-v1.1.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching_eye-v1.1.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching_lip-v1.1.onnx
