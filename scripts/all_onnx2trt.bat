@echo off

REM warping+spade model
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\warping_spade-fix.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\warping_spade-fix.onnx

REM landmark model
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\landmark.onnx

REM motion_extractor model
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\motion_extractor.onnx -p fp32
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\motion_extractor.onnx -p fp32

REM face_analysis model
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\retinaface_det_static.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\face_2dpose_106_static.onnx

REM appearance_extractor model
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\appearance_feature_extractor.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\appearance_feature_extractor.onnx

REM stitching model
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching_eye.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching_lip.onnx

.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\stitching.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\stitching_eye.onnx
.\venv\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\stitching_lip.onnx
