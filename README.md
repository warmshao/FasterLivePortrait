## FasterLivePortrait: Bring portraits to life in Real Time!
<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

**Original repository: [LivePortrait](https://github.com/KwaiVGI/LivePortrait), thanks to the authors for sharing**

**New features:**
* Achieved real-time running of LivePortrait on RTX 3090 GPU using TensorRT, reaching speeds of 30+ FPS.
* Implemented conversion of LivePortrait model to Onnx model, achieving inference speed of about 70ms/frame (~12 FPS) using onnxruntime-gpu on RTX 3090, facilitating cross-platform deployment.
* Seamless support for native gradio app, with several times faster speed and support for simultaneous inference on multiple faces. Some results can be seen here: [pr105](https://github.com/KwaiVGI/LivePortrait/pull/105)
* Refactored code structure, no longer dependent on pytorch, all models use onnx or tensorrt for inference.

**If you find this project useful, please give it a star ❤️❤️**

**Note: The above results were tested on Linux + RTX3090**

### Environment Setup
* `pip install -r requirements.txt`
* Install any additional missing dependencies as needed

### Onnxruntime Inference
* First, download the converted onnx model files from this address and place them in the `checkpoints` folder.
* If you want to use onnxruntime cpu inference, simply `pip install onnxruntime`. However, cpu inference is extremely slow and not recommended. Use GPU if possible.
* The latest onnxruntime-gpu still doesn't support grid_sample cuda, but I found a branch that supports it. Follow these steps to install `onnxruntime-gpu` from source:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks to liqun for the grid_sample with cuda implementation!
  * Run the following commands to compile, replacing cuda_version with your own:
  ```shell
  ./build.sh --parallel \
  --build_shared_lib --use_cuda \
  --cuda_version 11.8 \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/ \
  --config Release --build_wheel --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" \
  --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  --allow_running_as_root
  ```
  * `pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl`
* Test the pipeline using onnxruntime:
    ```
      python run.py \
     --src_image assets/examples/source/s10.jpg \
     --dri_video assets/examples/driving/d14.mp4 \
     --cfg configs/onnx_infer.yaml
     ```
### TensorRT Inference
* It's assumed that you have already installed TensorRT. If not, please Google for installation instructions. My TensorRT version is 8.6.1. Remember the installation path of TensorRT.
* Install the grid_sample tensorrt plugin, as the model requires 5d input for grid sample, which is not supported by the native grid_sample operator.
  * `git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin`
  * Modify line 30 in `CMakeLists.txt` to: `set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")`
  * `export PATH=/usr/local/cuda/bin:$PATH`
  * `mkdir build && cd build`
  * `cmake .. -DTensorRT_ROOT=$TENSORRT_HOME`, replace $TENSORRT_HOME with your TensorRT root directory.
  * `make`, remember the address of the so file, e.g., `/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so`
* Convert onnx models to tensorrt, replace the path in line 35 of `scripts/onnx2trt.py` with your own so path.
  * warping+spade model: `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/warping_spade-fix.onnx`
  * landmark model: `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/landmark.onnx`
  * motion_extractor model (note: this model can only use fp32 to avoid precision overflow): `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/motion_extractor.onnx -p fp32`
  * face_analysis model: `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/retinaface_det.onnx` and `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/face_2dpose_106.onnx`
  * appearance_extractor model: `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx`
  * stitching model:
    * `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching.onnx`
    * `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_eye.onnx`
    * `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_lip.onnx`
* Test the pipeline using tensorrt:
  ```shell
   python run.py \
   --src_image assets/examples/source/s10.jpg \
   --dri_video assets/examples/driving/d14.mp4 \
   --cfg configs/trt_infer.yaml
* To run in real-time using a camera:
  ```shell
   python run.py \
   --src_image assets/examples/source/s10.jpg \
   --dri_video assets/examples/driving/d14.mp4 \
   --cfg configs/trt_infer.yaml \
   --realtime
  ```
### Gradio App
* onnxruntime: `python app.py --mode onnx`
* tensorrt: `python app.py --mode trt`

### 关于我
Follow my shipinhao channel for continuous updates on my AIGC content. Feel free to message me for collaboration opportunities.
<img src="assets/shipinhao.jpg" alt="视频号" width="300" height="350">