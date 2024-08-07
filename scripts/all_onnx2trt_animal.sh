#!/bin/bash

# warping+spade model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/warping_spade-fix.onnx
# motion_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/motion_extractor.onnx -p fp32
# appearance_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/appearance_feature_extractor.onnx
# stitching model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching_eye.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_animal_onnx/stitching_lip.onnx
