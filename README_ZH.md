## FasterLivePortrait：Bring portrait to life in Real Time!
<a href="README.md">English</a> | <a href="README_ZH.md">中文</a>

**原仓库: [LivePortrait](https://github.com/KwaiVGI/LivePortrait)，感谢作者的分享**

**新增功能：**
* 通过TensorRT实现在RTX 3090显卡上**实时**运行LivePortrait，速度达到 30+ FPS. 这个速度是实测渲染出一帧的速度，而不仅仅是模型的推理时间。
* 无缝支持原生的gradio app, 速度快了好几倍，同时支持多张人脸、Animal模型。
* 增加[JoyVASA](https://github.com/jdh-algo/JoyVASA)的支持，可以用音频驱动视频或图片。

**如果你觉得这个项目有用，帮我点个star吧✨✨**

### Demo(还有很多功能等你探索)
* 声音驱动视频(可以实时):

<video src="https://github.com/user-attachments/assets/98bb5ff7-0796-42db-9d7b-e04ddd2c3c14" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>
* 动物驱动:

<video src="https://github.com/user-attachments/assets/dada0a92-593a-480b-a034-cbcce16e38b9" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>
* 多张人脸同时驱动:

<video src="https://github.com/KwaiVGI/LivePortrait/assets/138360003/b37de35d-6feb-4100-b73f-58ac23121483" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>


### 环境安装
* 方式1：如果你是Windows用户，推荐可以直接下载[整合包](https://github.com/warmshao/FasterLivePortrait/releases/tag/v1.8)。
* 方式2：Docker，提供了一个镜像，不用再自己安装onnxruntime-gpu和TensorRT。
  * 根据自己的系统安装[docker](https://docs.docker.com/desktop/install/windows-install/)
  * 下载镜像：`docker pull shaoguo/faster_liveportrait:v3`
  * 执行命令, `$FasterLivePortrait_ROOT`要替换成你下载的FasterLivePortrait在本地的目录:
  ```shell
  docker run -it --gpus=all \
  --name faster_liveportrait \
  -v $FasterLivePortrait_ROOT:/root/FasterLivePortrait \
  --restart=always \
  -p 9870:9870 \
  shaoguo/faster_liveportrait:v3 \
  /bin/bash
  ```
  * 然后可以根据下面Onnxruntime 推理和TensorRT 推理教程进行使用。
  
* 方式3：新建一个python虚拟环境，自己安装必要的python包
  * 请先安装[ffmpeg](https://www.ffmpeg.org/download.html)
  * `pip install -r requirements.txt`
  * 再根据以下教程安装onnxruntime-gpu或TensorRT。

### 使用方法
#### 1. TensorRT 推理(推荐, 可以实时)
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
#### 2. Onnxruntime 推理
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

### Gradio WebUI
* onnxruntime: `python webui.py --mode onnx`
* tensorrt: `python webui.py --mode trt`
* 默认端口在9870，打开网页：`http://localhost:9870/`

**日志**
- [x] **2024/12/22:** 增加api部署`python api.py`, 其他参考[教程](assets/docs/API_ZH.md)使用。
- [x] **2024/12/16:** 增加[JoyVASA](https://github.com/jdh-algo/JoyVASA)的支持，可以用音频驱动视频或图片。非常酷！
  - 更新代码，然后下载模型: `huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir .\checkpoints\chinese-hubert-base` 和 ` huggingface-cli download jdh-algo/JoyVASA --local-dir ./checkpoints/JoyVASA`
  - 启动webui后根据以下教程使用即可，建议source 是视频的情况下只驱动嘴部

   <video src="https://github.com/user-attachments/assets/42fb24be-0cde-4138-9671-e52eec95e7f5" controls="controls" width="500" height="400">您的浏览器不支持播放该视频！</video>
  
- [x] **2024/12/14:** 增加pickle和image驱动以及区域驱动`animation_region`。
  - 请更新最新的代码，windows用户可以直接双击`update.bat`更新，但请注意本地的代码将会被覆盖。
  - `python run.py ` 现在运行 `driving video`会自动保存对应的pickle到跟`driving video`一样的目录，可以直接复用。
  - 打开`webui`后即可体验新的pickle和image驱动以及区域驱动`animation_region`等功能。注意image驱动记得把`relative motion`取消掉。
- [x] **2024/08/11:** 优化paste_back的速度，修复一些bug。
  - 用torchgeometry + cuda优化paste_back函数，现在速度提升了很多。示例：`python run.py --src_image assets/examples/source/s39.jpg --dri_video assets/examples/driving/d0.mp4 --cfg configs/trt_infer.yaml --paste_back --animal`
  - 修复Xpose的ops在一些显卡运行报错的问题等bug。请使用最新的镜像:`docker pull shaoguo/faster_liveportrait:v3`
- [x] **2024/08/07:** 增加animal模型的支持，同时支持mediapipe模型，现在你不用再担心版权的问题。
  - 增加对animal模型的支持。
    - 需要下载animal的onnx文件：`huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`，然后转换成trt文件。
    - 更新镜像`docker pull shaoguo/faster_liveportrait:v3`, 使用animal模型的示例:`python run.py --src_image assets/examples/source/s39.jpg --dri_video 0 --cfg configs/trt_infer.yaml --realtime --animal`
    - windows系统可以从release页下载最新的[windows 整合包](https://github.com/warmshao/FasterLivePortrait/releases)，解压后使用。
    - 简单的使用教程：
    
    <video src="https://github.com/user-attachments/assets/dc37e2dd-551a-43b0-8929-fc5d5fe16ec5" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>
    
  - 使用mediapipe模型替代insight_face
    - 网页端使用: `python webui.py --mode trt --mp` 或 `python webui.py --mode onnx --mp`
    - 本地摄像头运行: `python run.py --src_image assets/examples/source/s12.jpg --dri_video assets/examples/driving/d0.mp4 --cfg configs/trt_mp_infer.yaml`
- [x] **2024/07/24:** Windows的整合包, 免安装一键运行，支持TensorRT和OnnxruntimeGPU。感谢@zhanghongyong123456在[issue](https://github.com/warmshao/FasterLivePortrait/issues/22)的贡献。
  - 【可选】如果你的windows电脑已经装过cuda和cudnn，请忽略这一步。我只在cuda12.2上验证过，如果没安装cuda或报cuda相关的错，你需要按照以下步骤进行安装：
    - 下载[cuda12.2](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Windows&target_arch=x86_64), 双击exe后按照默认设置一步步安装即可。
    - 下载[cudnn](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip) 压缩包，解压后将cudnn 文件夹下的lib、bin、include 文件夹复制到 CUDA12.2 文件夹下（默认为C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2）
  - 从release页下载免安装[windows 整合包](https://github.com/warmshao/FasterLivePortrait/releases)并解压。
  - 进入`FasterLivePortrait-windows`后双击`scripts/all_onnx2trt.bat`对onnx文件进行转换，这会等上一段时间。
  - 网页端demo：双击`webui.bat`, 打开网页：`http://localhost:9870/`
  - 摄像头实时运行，双击`camera.bat`，按`q`停止。如果你想更换目标图像，命令行运行:`camera.bat assets/examples/source/s9.jpg`。
- [x] **2024/07/18:** MacOS支持(不需要Docker，python就可以了），M1/M2的速度比较快，但还是很慢😟
  - 安装ffmpeg: `brew install ffmpeg`
  - 安装python=3.10的虚拟环境，推荐可以用[miniforge](https://github.com/conda-forge/miniforge).`conda create -n flip python=3.10 && conda activate flip`
  - `pip install -r requirements_macos.txt`
  - 下载onnx文件: `huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints`
  - 测试: `python webui.py --mode onnx`
- [x] **2024/07/17:** 增加docker环境的支持，提供可运行的镜像。