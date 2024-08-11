## FasterLivePortrait: Bring portraits to life in Real Time!
<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

**Original repository: [LivePortrait](https://github.com/KwaiVGI/LivePortrait), thanks to the authors for sharing**

**New features:**
* Achieved real-time running of LivePortrait on RTX 3090 GPU using TensorRT, reaching speeds of 30+ FPS. This is the speed for rendering a single frame, including pre- and post-processing, not just the model inference speed.
* Implemented conversion of LivePortrait model to Onnx model, achieving inference speed of about 70ms/frame (~12 FPS) using onnxruntime-gpu on RTX 3090, facilitating cross-platform deployment.
* Seamless support for native gradio app, with several times faster speed and support for simultaneous inference on multiple faces and Animal Model.

**If you find this project useful, please give it a star ✨✨**

<video src="https://github.com/user-attachments/assets/dada0a92-593a-480b-a034-cbcce16e38b9" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

<video src="https://github.com/user-attachments/assets/716d61a7-41ae-483a-874d-ea1bf345bd1a" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

**Changelog**
- [x] **2024/08/11:** Optimized paste_back speed and fixed some bugs.
  - Used torchgeometry + cuda to optimize the paste_back function, significantly improving speed. Example: `python run.py --src_image assets/examples/source/s39.jpg --dri_video assets/examples/driving/d0.mp4 --cfg configs/trt_infer.yaml --paste_back --animal`
  - Fixed issues with Xpose ops causing errors on some GPUs and other bugs. Please use the latest docker image: `docker pull shaoguo/faster_liveportrait:v3`
- [x] **2024/08/07:** Added support for animal models and MediaPipe models, so you no longer need to worry about copyright issues.
  - Added support for animal models.
    - Download the animal ONNX file: `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`, then convert it to TRT format.
    - Update the Docker image: `docker pull shaoguo/faster_liveportrait:v3`. Using animal model:`python run.py --src_image assets/examples/source/s39.jpg --dri_video 0 --cfg configs/trt_infer.yaml --realtime --animal`
    - Windows users can download the latest [Windows all-in-one package](https://github.com/warmshao/FasterLivePortrait/releases) from the release page, then unzip and use it.
    - Simple usage tutorial:
    
    <video src="https://github.com/user-attachments/assets/dc37e2dd-551a-43b0-8929-fc5d5fe16ec5" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>
    
  - Using MediaPipe model to replace InsightFace
    - For web usage: `python app.py --mode trt --mp` or `python app.py --mode onnx --mp`
    - For local webcam: `python run.py --src_image assets/examples/source/s12.jpg --dri_video 0 --cfg configs/trt_mp_infer.yaml`
- [x] **2024/07/24:** Windows integration package, no installation required, one-click run, supports TensorRT and OnnxruntimeGPU. Thanks to @zhanghongyong123456 for their contribution in this [issue](https://github.com/warmshao/FasterLivePortrait/issues/22).
  - [Optional] If you have already installed CUDA and cuDNN on your Windows computer, please skip this step. I have only verified on CUDA 12.2. If you haven't installed CUDA or encounter CUDA-related errors, you need to follow these steps:
    - Download [CUDA 12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64), double-click the exe and install following the default settings step by step.
    - Download the [cuDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip) zip file, extract it, and copy the lib, bin, and include folders from the cuDNN folder to the CUDA 12.2 folder (default is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2)
  - Download the installation-free [Windows integration package](https://github.com/warmshao/FasterLivePortrait/releases) from the release page and extract it.
  - Enter `FasterLivePortrait-windows` and double-click `all_onnx2trt.bat` to convert onnx files, which will take some time.
  - For web demo: Double-click `app.bat`, open the webpage: `http://localhost:9870/`
  - For real-time camera operation, double-click `camera.bat`，press `q` to stop. If you want to change the target image, run in command line: `camera.bat assets/examples/source/s9.jpg`
- [x] **2024/07/18:** macOS support added(No need for Docker, Python is enough). M1/M2 chips are faster, but it's still quite slow 😟
  - Install ffmpeg: `brew install ffmpeg`
  - Set up a Python 3.10 virtual environment. Recommend using [miniforge](https://github.com/conda-forge/miniforge): `conda create -n flip python=3.10 && conda activate flip`
  - Install requirements: `pip install -r requirements_macos.txt`
  - Download ONNX files: `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`
  - Test: `python app.py --mode onnx`
- [x] **2024/07/17:** Added support for Docker environment, providing a runnable image.

### Environment Setup
* Option 1: Docker (recommended).A docker image is provided for  eliminating the need to install onnxruntime-gpu and TensorRT manually.
  * Install [Docker](https://docs.docker.com/desktop/install/windows-install/) according to your system
  * Download the image: `docker pull shaoguo/faster_liveportrait:v3`
  * Execute the command, replace `$FasterLivePortrait_ROOT` with the local directory where you downloaded FasterLivePortrait:
  ```shell
  docker run -it --gpus=all \
  --name faster_liveportrait \
  -v $FasterLivePortrait_ROOT:/root/FasterLivePortrait \
  --restart=always \
  -p 9870:9870 \
  shaoguo/faster_liveportrait:v3 \
  /bin/bash
  ```
* Option 2: Create a new Python virtual environment and install the necessary Python packages manually.
  * First, install [ffmpeg](https://www.ffmpeg.org/download.html)
  * Run `pip install -r requirements.txt`
  * Then follow the tutorials below to install onnxruntime-gpu or TensorRT. Note that this has only been tested on Linux systems.

### Onnxruntime Inference
* First, download the converted onnx model files:`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`.
* (Ignored in Docker)If you want to use onnxruntime cpu inference, simply `pip install onnxruntime`. However, cpu inference is extremely slow and not recommended. The latest onnxruntime-gpu still doesn't support grid_sample cuda, but I found a branch that supports it. Follow these steps to install `onnxruntime-gpu` from source:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks to liqun for the grid_sample with cuda implementation!
  * Run the following commands to compile, changing `cuda_version` and `CMAKE_CUDA_ARCHITECTURES` according to your machine:
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
* (Ignored in Docker) Install TensorRT. Remember the installation path of [TensorRT](https://developer.nvidia.com/tensorrt).
* (Ignored in Docker) Install the grid_sample TensorRT plugin, as the model uses grid sample that requires 5D input, which is not supported by the native grid_sample operator.
  * `git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin`
  * Modify line 30 in `CMakeLists.txt` to: `set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")`
  * `export PATH=/usr/local/cuda/bin:$PATH`
  * `mkdir build && cd build`
  * `cmake .. -DTensorRT_ROOT=$TENSORRT_HOME`, replace $TENSORRT_HOME with your own TensorRT root directory.
  * `make`, remember the address of the .so file, replace `/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so` in `scripts/onnx2trt.py` and `src/models/predictor.py` with your own .so file path
* Download ONNX model files:`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`. Convert all ONNX models to TensorRT, run `sh scripts/all_onnx2trt.sh` and `sh scripts/all_onnx2trt_animal.sh`
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
   --dri_video 0 \
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