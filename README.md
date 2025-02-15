## FasterLivePortrait: Bring portraits to life in Real Time!
<a href="README.md">English</a> | <a href="README_ZH.md">中文</a>

**Original repository: [LivePortrait](https://github.com/KwaiVGI/LivePortrait), thanks to the authors for sharing**

**New features:**
* Achieved real-time running of LivePortrait on RTX 3090 GPU using TensorRT, reaching speeds of 30+ FPS. This is the speed for rendering a single frame, including pre- and post-processing, not just the model inference speed.
* Seamless support for native gradio app, with several times faster speed and support for simultaneous inference on multiple faces and Animal Model.
* Added support for [JoyVASA](https://github.com/jdh-algo/JoyVASA), which can drive videos or images with audio.

**If you find this project useful, please give it a star ✨✨**

### Demo (Explore more features)
* Text-driven video, based on kokoro-82M:

<video src="https://github.com/user-attachments/assets/04e962e2-6c57-4d01-ae4a-2f6d2d501c5a" controls="controls" width="500" height="300">Your browser does not support this video!</video>

* Audio-driven video (real-time):

<video src="https://github.com/user-attachments/assets/98bb5ff7-0796-42db-9d7b-e04ddd2c3c14" controls="controls" width="500" height="300">Your browser does not support this video!</video>

* Animal-driven:

<video src="https://github.com/user-attachments/assets/dada0a92-593a-480b-a034-cbcce16e38b9" controls="controls" width="500" height="300">Your browser does not support this video!</video>

* Multiple faces driven simultaneously:

<video src="https://github.com/KwaiVGI/LivePortrait/assets/138360003/b37de35d-6feb-4100-b73f-58ac23121483" controls="controls" width="500" height="300">Your browser does not support this video!</video>


### Environment Setup
* Option 1 (recommended): If you are a Windows user, you can directly download the [integrated package](https://github.com/warmshao/FasterLivePortrait/releases/tag/v1.8).
    * You need to install [git](https://git-scm.com/downloads) first, then double-click `update.bat` to update the code.
    * Double-click `scripts/all_onnx2trt.bat` to convert onnx files to tensorrt files.
    * Double-click `webui.bat` to open the webpage, or double-click `camera.bat` to open the camera for real-time operation.
* Option 2: Docker.A docker image is provided for  eliminating the need to install onnxruntime-gpu and TensorRT manually.
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
* Option 3: Create a new Python virtual environment and install the necessary Python packages manually.
  * First, install [ffmpeg](https://www.ffmpeg.org/download.html)
  * Run `pip install -r requirements.txt`
  * Then follow the tutorials below to install onnxruntime-gpu or TensorRT. Note that this has only been tested on Linux systems.

### Usage
#### 1. TensorRT Inference(Recommended)
* (Ignored in Docker) Install TensorRT 8.x (versions >=10.x are not compatible). Remember the installation path of [TensorRT](https://developer.nvidia.com/tensorrt).
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
  
#### 2. Onnxruntime Inference
* First, download the converted onnx model files:`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`.
* (Ignored in Docker)If you want to use onnxruntime cpu inference, simply `pip install onnxruntime`. However, cpu inference is extremely slow and not recommended. The latest onnxruntime-gpu still doesn't support grid_sample cuda, but I found a branch that supports it. Follow these steps to install `onnxruntime-gpu` from source:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks to liqun for the grid_sample with cuda implementation!
  * Run the following commands to compile, changing `cuda_version` and `CMAKE_CUDA_ARCHITECTURES` according to your machine (your cuDNN version must be 8.x, 9.x is not compatible):
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

### Gradio WebUI
* onnxruntime: `python webui.py --mode onnx`
* tensorrt: `python webui.py --mode trt`
* The default port is 9870. Open the webpage: `http://localhost:9870/`


Hotkeys for webcam mode (when render window is on focus)\
Q > exit\
S > Stitching\
Z > RelativeMotion\
X > AnimationRegion\
C > CropDrivingVideo\
K,L > AdjustSourceScale\
N,M > AdjustDriverScale


**Changelog**
- [x] **2024/12/22:** Add API Deployment `python api.py`, For more information, please refer to the [tutorial](assets/docs/API.md).
- [x] **2024/12/21:** Added support for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), enabling text-driven video or image generation.
  - Updated code: `git pull origin master` and install the latest Python dependencies `pip install requirements.txt`, or simply double-click `update.bat` on Windows.
  - Download the model: `huggingface-cli download hexgrad/Kokoro-82M --local-dir .\checkpoints\Kokoro-82M`.
  - For Linux, install `espeak-ng`: `apt-get -qq -y install espeak-ng > /dev/null 2>&1`
  - For Windows, refer to [manual installation instructions](https://huggingface.co/hexgrad/Kokoro-82M/discussions/12) and configure the `espeak-ng` environment variables.  The current read location is [here](src/pipelines/gradio_live_portrait_pipeline.py:437); modify it if your installation path differs.
  -  Now you can use it normally in the "Drive Text" tab.
- [x] **2024/12/16:** Added support for [JoyVASA](https://github.com/jdh-algo/JoyVASA), which can drive videos or images with audio. Very cool!
 - Update code, then download the models: `huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir .\checkpoints\chinese-hubert-base` and `huggingface-cli download jdh-algo/JoyVASA --local-dir ./checkpoints/JoyVASA`
 - After launching the webui, follow the tutorial below. When the source is a video, it's recommended to only drive the mouth movements
  
  <video src="https://github.com/user-attachments/assets/42fb24be-0cde-4138-9671-e52eec95e7f5" controls="controls" width="500" height="400">您的浏览器不支持播放该视频！</video>

- [x] **2024/12/14:** Added pickle and image driving, as well as region driving animation_region.
  - Please update the latest code. Windows users can directly double-click `update.bat` to update, but note that your local code will be overwritten.
  - Running `python run.py` now automatically saves the corresponding pickle to the same directory as the driving video, allowing for direct reuse.
  - After opening webui, you can experience the new pickle and image driving, as well as the region driving animation_region features. Note that for image driving, remember to disable `relative motion`.
- [x] **2024/08/11:** Optimized paste_back speed and fixed some bugs.
  - Used torchgeometry + cuda to optimize the paste_back function, significantly improving speed. Example: `python run.py --src_image assets/examples/source/s39.jpg --dri_video assets/examples/driving/d0.mp4 --cfg configs/trt_infer.yaml --paste_back --animal`
  - Fixed issues with Xpose ops causing errors on some GPUs and other bugs. Please use the latest docker image: `docker pull shaoguo/faster_liveportrait:v3`
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
    - For web usage: `python webui.py --mode trt --mp` or `python webui.py --mode onnx --mp`
    - For local webcam: `python run.py --src_image assets/examples/source/s12.jpg --dri_video 0 --cfg configs/trt_mp_infer.yaml`
- [x] **2024/07/24:** Windows integration package, no installation required, one-click run, supports TensorRT and OnnxruntimeGPU. Thanks to @zhanghongyong123456 for their contribution in this [issue](https://github.com/warmshao/FasterLivePortrait/issues/22).
  - [Optional] If you have already installed CUDA and cuDNN on your Windows computer, please skip this step. I have only verified on CUDA 12.2. If you haven't installed CUDA or encounter CUDA-related errors, you need to follow these steps:
    - Download [CUDA 12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64), double-click the exe and install following the default settings step by step.
    - Download the [cuDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip) zip file, extract it, and copy the lib, bin, and include folders from the cuDNN folder to the CUDA 12.2 folder (default is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2)
  - Download the installation-free [Windows integration package](https://github.com/warmshao/FasterLivePortrait/releases) from the release page and extract it.
  - Enter `FasterLivePortrait-windows` and double-click `scripts/all_onnx2trt.bat` to convert onnx files, which will take some time.
  - For web demo: Double-click `webui.bat`, open the webpage: `http://localhost:9870/`
  - For real-time camera operation, double-click `camera.bat`，press `q` to stop. If you want to change the target image, run in command line: `camera.bat assets/examples/source/s9.jpg`
- [x] **2024/07/18:** macOS support added(No need for Docker, Python is enough). M1/M2 chips are faster, but it's still quite slow 😟
  - Install ffmpeg: `brew install ffmpeg`
  - Set up a Python 3.10 virtual environment. Recommend using [miniforge](https://github.com/conda-forge/miniforge): `conda create -n flip python=3.10 && conda activate flip`
  - Install requirements: `pip install -r requirements_macos.txt`
  - Download ONNX files: `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`
  - Test: `python webui.py --mode onnx`
- [x] **2024/07/17:** Added support for Docker environment, providing a runnable image.
