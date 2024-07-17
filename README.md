## FasterLivePortrait: Bring portraits to life in Real Time!
<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

**Original repository: [LivePortrait](https://github.com/KwaiVGI/LivePortrait), thanks to the authors for sharing**

**New features:**
* Achieved real-time running of LivePortrait on RTX 3090 GPU using TensorRT, reaching speeds of 30+ FPS. This is the speed for rendering a single frame, including pre- and post-processing, not just the model inference speed.
* Implemented conversion of LivePortrait model to Onnx model, achieving inference speed of about 70ms/frame (~12 FPS) using onnxruntime-gpu on RTX 3090, facilitating cross-platform deployment.
* Seamless support for native gradio app, with several times faster speed and support for simultaneous inference on multiple faces. Some results can be seen here: [pr105](https://github.com/KwaiVGI/LivePortrait/pull/105)
* Refactored code structure, no longer dependent on pytorch, all models use onnx or tensorrt for inference.

**If you find this project useful, please give it a star ❤️❤️**

<video src="https://github.com/KwaiVGI/LivePortrait/assets/138360003/c0c8de4f-6a6f-43fa-89f9-168ff3f150ef" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

**Changelog**
- [x] **2024/07/17:** Added support for Docker environment, providing a runnable image.
- [ ] Windows integration package, supports one-click run
- [ ] MacOS integration package, supports one-click run

### Environment Setup
* Option 1: Docker (recommended).A docker image is provided for  eliminating the need to install onnxruntime-gpu and TensorRT manually.
  * Install [Docker](https://docs.docker.com/desktop/install/windows-install/) according to your system
  * Download the image: `docker pull shaoguo/faster_liveportrait:v1`
  * Execute the command, replace `$FasterLivePortrait_ROOT` with the local directory where you downloaded FasterLivePortrait:
  ```shell
  docker run -it --gpus=all \
  --name faster_liveportrait \
  -v E:\\data:/data \
  -v $FasterLivePortrait_ROOT:/root/FasterLivePortrait \
  --restart=always \
  -p 9870:9870 \
  shaoguo/faster_liveportrait:v1 \
  /bin/bash
  ```
* Option 2: Create a new Python virtual environment and install the necessary Python packages manually
  * First, install [ffmpeg](https://www.ffmpeg.org/download.html)
  * Run `pip install -r requirements.txt`
  * Then follow the tutorials below to install onnxruntime-gpu or TensorRT.

### Onnxruntime Inference
* First, download the converted onnx model files from [this](https://huggingface.co/warmshao/FasterLivePortrait) and place them in the `checkpoints` folder.
* If you want to use onnxruntime cpu inference, simply `pip install onnxruntime`. However, cpu inference is extremely slow and not recommended. Use GPU if possible.
* The latest onnxruntime-gpu still doesn't support grid_sample cuda, but I found a branch that supports it. Follow these steps to install `onnxruntime-gpu` from source:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks to liqun for the grid_sample with cuda implementation!
  * Run the following commands to compile
  ```shell
  ./build.sh --parallel \
  --build_shared_lib --use_cuda \
  --cuda_version 11.8 \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/ \
  --config Release --build_wheel --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="60;70;75;80;86" \
  --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  --disable_contrib_ops \
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
    * add `include_directories("/usr/local/cuda/include")` in `CMakeLists.txt` if `lib_cuda.h` not found.
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
* The default port is 9870. Open the webpage: `http://localhost:9870/`

### About Me
Follow my shipinhao channel for continuous updates on my AIGC content. Feel free to message me for collaboration opportunities.

<img src="assets/shipinhao.jpg" alt="视频号" width="300" height="350">