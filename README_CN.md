## FasterLivePortrait：Bring portrait to life in Real Time!
<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

**原仓库: [LivePortrait](https://github.com/KwaiVGI/LivePortrait)，感谢作者的分享**

**新增功能：**
* 通过TensorRT实现在RTX 3090显卡上**实时**运行LivePortrait，速度达到 30+ FPS. 这个速度是实测渲染出一帧的速度，而不仅仅是模型的推理时间。
* 实现将LivePortrait模型转为Onnx模型，使用onnxruntime-gpu在RTX 3090上的推理速度约为 70ms/帧（～12 FPS），方便跨平台的部署。
* 无缝支持原生的gradio app, 速度快了好几倍，同时支持对多张人脸的同时推理，一些效果可以看：[pr105](https://github.com/KwaiVGI/LivePortrait/pull/105)
* 对代码结构进行了重构，不再依赖pytorch，所有的模型用onnx或tensorrt推理。

**如果你觉得这个项目有用，帮我点个star吧✨✨**

<video src="https://github.com/user-attachments/assets/dada0a92-593a-480b-a034-cbcce16e38b9" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

<video src="https://github.com/user-attachments/assets/716d61a7-41ae-483a-874d-ea1bf345bd1a" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

**日志**
- [x] **2024/08/07:** 增加animal模型的支持，同时支持mediapipe模型，现在你不用再担心版权的问题。
  - 增加对animal模型的支持。
    - 需要下载animal的onnx文件：`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`，然后转换成trt文件。
    - 更新镜像`docker pull shaoguo/faster_liveportrait:v2`, 使用animal模型的示例:`python run.py --src_image assets/examples/source/s39.jpg --dri_video 0 --cfg configs/trt_infer.yaml --realtime --animal`
    - windows系统可以从release页下载最新的[windows 整合包](https://github.com/warmshao/FasterLivePortrait/releases)，解压后使用。
    - 简单的使用教程：
    
    <video src="https://github.com/user-attachments/assets/dc37e2dd-551a-43b0-8929-fc5d5fe16ec5" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>
    
  - 使用mediapipe模型替代insight_face
    - 网页端使用: `python app.py --mode trt --mp` 或 `python app.py --mode onnx --mp`
    - 本地摄像头运行: `python run.py --src_image assets/examples/source/s12.jpg --dri_video assets/examples/driving/d0.mp4 --cfg configs/trt_mp_infer.yaml`
- [x] **2024/07/24:** Windows的整合包, 免安装一键运行，支持TensorRT和OnnxruntimeGPU。感谢@zhanghongyong123456在[issue](https://github.com/warmshao/FasterLivePortrait/issues/22)的贡献。
  - 【可选】如果你的windows电脑已经装过cuda和cudnn，请忽略这一步。我只在cuda12.2上验证过，如果没安装cuda或报cuda相关的错，你需要按照以下步骤进行安装：
    - 下载[cuda12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64), 双击exe后按照默认设置一步步安装即可。
    - 下载[cudnn](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip) 压缩包，解压后将cudnn 文件夹下的lib、bin、include 文件夹复制到 CUDA12.2 文件夹下（默认为C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2）
  - 从release页下载免安装[windows 整合包](https://github.com/warmshao/FasterLivePortrait/releases)并解压。
  - 进入`FasterLivePortrait-windows`后双击`all_onnx2trt.bat`对onnx文件进行转换，这会等上一段时间。
  - 网页端demo：双击`app.bat`, 打开网页：`http://localhost:9870/`
  - 摄像头实时运行，双击`camera.bat`，按`q`停止。如果你想更换目标图像，命令行运行:`camera.bat assets/examples/source/s9.jpg`。
- [x] **2024/07/18:** MacOS支持(不需要Docker，python就可以了），M1/M2的速度比较快，但还是很慢😟
  - 安装ffmpeg: `brew install ffmpeg`
  - 安装python=3.10的虚拟环境，推荐可以用[miniforge](https://github.com/conda-forge/miniforge).`conda create -n flip python=3.10 && conda activate flip`
  - `pip install -r requirements_macos.txt`
  - 下载onnx文件: `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`
  - 测试: `python app.py --mode onnx`
- [x] **2024/07/17:** 增加docker环境的支持，提供可运行的镜像。


### 环境安装
* 方式1：Docker(推荐），提供了一个镜像，不用再自己安装onnxruntime-gpu和TensorRT。
  * 根据自己的系统安装[docker](https://docs.docker.com/desktop/install/windows-install/)
  * 下载镜像：`docker pull shaoguo/faster_liveportrait:v2`
  * 执行命令, `$FasterLivePortrait_ROOT`要替换成你下载的FasterLivePortrait在本地的目录:
  ```shell
  docker run -it --gpus=all \
  --name faster_liveportrait \
  -v $FasterLivePortrait_ROOT:/root/FasterLivePortrait \
  --restart=always \
  -p 9870:9870 \
  shaoguo/faster_liveportrait:v1 \
  /bin/bash
  ```
  * 然后可以根据下面Onnxruntime 推理和TensorRT 推理教程进行使用。
  
* 方式2：新建一个python虚拟环境，自己安装必要的python包
  * 请先安装[ffmpeg](https://www.ffmpeg.org/download.html)
  * `pip install -r requirements.txt`
  * 再根据以下教程安装onnxruntime-gpu或TensorRT，注意只有在Linux系统下实验过。

### Onnxruntime 推理
* 首先下载我转换好的[模型onnx文件](https://huggingface.co/warmshao/FasterLivePortrait): `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`。
* (Docker环境可忽略）如果你要用onnxruntime cpu推理的话，直接`pip install onnxruntime`即可，但是cpu推理超级慢。但是最新的onnxruntime-gpu仍然无法支持grid_sample cuda，好在我看到一位大佬在分支上支持了，按照以下步骤源码安装`onnxruntime-gpu`:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks for liqun's grid_sample with cuda implementation!
  * 运行以下命令编译,`cuda_version`和`CMAKE_CUDA_ARCHITECTURES`根据自己的机器更改:
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
  * `pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl`就可以了
* 用onnxruntime测试pipeline：
  ```shell
   python run.py \
   --src_image assets/examples/source/s10.jpg \
   --dri_video assets/examples/driving/d14.mp4 \
   --cfg configs/onnx_infer.yaml
  ```
  
### TensorRT 推理
* (Docker环境可忽略）安装TensorRT，请记住[TensorRT](https://developer.nvidia.com/tensorrt)安装的路径。
* (Docker环境可忽略）安装 grid_sample的tensorrt插件，因为模型用到的grid sample需要有5d的输入,原生的grid_sample 算子不支持。
  * `git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin`
  * 修改`CMakeLists.txt`中第30行为:`set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")`
  * `export PATH=/usr/local/cuda/bin:$PATH`
  * `mkdir build && cd build`
  * `cmake .. -DTensorRT_ROOT=$TENSORRT_HOME`,$TENSORRT_HOME 替换成你自己TensorRT的根目录。
  * `make`，记住so文件的地址，将`scripts/onnx2trt.py`和`src/models/predictor.py`里`/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so`替换成自己的so路径
* 下载Onnx文件：`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`。将onnx模型转为tensorrt，运行`sh scripts/all_onnx2trt.sh`和`sh scripts/all_onnx2trt_animal.sh`
* 用tensorrt测试pipeline：
  ```shell
   python run.py \
   --src_image assets/examples/source/s10.jpg \
   --dri_video assets/examples/driving/d14.mp4 \
   --cfg configs/trt_infer.yaml
  ```
  如果要使用摄像头实时运行：
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
* 默认端口在9870，打开网页：`http://localhost:9870/`

### 关于我
欢迎关注我的视频号，会持续分享我做的AIGC的内容。有合作需求欢迎私信。

<img src="assets/shipinhao.jpg" alt="视频号" width="300" height="350">

