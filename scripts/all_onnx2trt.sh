#!/bin/bash

# warping+spade model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/warping_spade-fix.onnx
# landmark model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/landmark.onnx
# motion_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/motion_extractor.onnx -p fp32
# face_analysis model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/retinaface_det_static.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx
# appearance_extractor model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx
# stitching model
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_eye.onnx
python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_lip.onnx
