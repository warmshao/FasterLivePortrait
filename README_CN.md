## FasterLivePortrait：Bring portrait to life in Real Time!
<a href="README.md">English</a> | <a href="README_CN.md">中文</a>

**原仓库: [LivePortrait](https://github.com/KwaiVGI/LivePortrait)，感谢作者的分享**

**新增功能：**
* 通过TensorRT实现在RTX 3090显卡上**实时**运行LivePortrait，速度达到 30+ FPS.
* 实现将LivePortrait模型转为Onnx模型，使用onnxruntime-gpu在RTX 3090上的推理速度约为 70ms/帧（～12 FPS），方便跨平台的部署。
* 无缝支持原生的gradio app, 速度快了好几倍，同时支持对多张人脸的同时推理，一些效果可以看：[pr105](https://github.com/KwaiVGI/LivePortrait/pull/105)
* 对代码结构进行了重构，不再依赖pytorch，所有的模型用onnx或tensorrt推理。

**如果你觉得这个项目有用，帮我点个star吧❤️❤️**

<video src="https://github.com/KwaiVGI/LivePortrait/assets/138360003/c0c8de4f-6a6f-43fa-89f9-168ff3f150ef" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

**注意：以上结果是在Linux+RTX3090上做的测试**

### 环境安装
* `pip install -r requirements.txt`
* 缺啥你再自己安装吧

### Onnxruntime 推理
* 首先从[这里](https://huggingface.co/warmshao/FasterLivePortrait)下载我转换好的模型onnx文件，放在`checkpoints`文件夹下。
* 如果你要用onnxruntime cpu推理的话，直接`pip install onnxruntime`即可，但是cpu推理超级慢，这里不推荐。有条件还是得用GPU。
* 但是最新的onnxruntime-gpu仍然无法支持grid_sample cuda，好在我看到一位大佬在分支上支持了，按照以下步骤源码安装`onnxruntime-gpu`:
  * `git clone https://github.com/microsoft/onnxruntime`
  * `git checkout liqun/ImageDecoder-cuda`. Thanks for liqun's grid_sample with cuda implementation!
  * 运行以下命令编译,cuda_version要改成你自己的:
  ```shell
  build cmd:
  ./build.sh --parallel \
  --build_shared_lib --use_cuda \
  --cuda_version 11.8 \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/ \
  --config Release --build_wheel --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80" \
  --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
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
* 默认你已经安装过TensorRT，如果还没装好，请自行Google安装，不再赘述。我的TensorRT是8.6.1，另外记住TensorRT安装的路径。
* 安装 grid_sample的tensorrt插件，因为模型用到的grid sample需要有5d的输入,原生的grid_sample 算子不支持。
  * `git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin`
  * 修改`CMakeLists.txt`中第30行为:`set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86")`
  * `export PATH=/usr/local/cuda/bin:$PATH`
  * `mkdir build && cd build`
  * `cmake .. -DTensorRT_ROOT=$TENSORRT_HOME`,$TENSORRT_HOME 替换成你自己TensorRT的根目录。
  * `make`，记住so文件的地址，比如我的是在：`/opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so`
* 将onnx模型转为tensorrt，将`scripts/onnx2trt.py`里第35行路径替换成自己的so路径。
  * warping+spade model：`python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/warping_spade-fix.onnx`
  * landmark model: `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/landmark.onnx`
  * motion_extractor model, 注意这个模型只能用fp32, 不然会遇到精度溢出的问题：`python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/motion_extractor.onnx -p fp32`
  * face_analysis model: `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/retinaface_det.onnx` 和 `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/face_2dpose_106.onnx`
  * appearance_extractor model: ` python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx`
  * stitching model:
    * `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching.onnx`
    * `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_eye.onnx`
    * `python scripts/onnx2trt.py -o ./checkpoints/liveportrait_onnx/stitching_lip.onnx`
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
   --dri_video assets/examples/driving/d14.mp4 \
   --cfg configs/trt_infer.yaml \
   --realtime
  ```
### Gradio App
* onnxruntime: `python app.py --mode onnx`
* tensorrt: `python app.py --mode trt`

### 关于我
欢迎关注我的视频号，会持续分享我做的AIGC的内容。有合作需求欢迎私信。

<img src="assets/shipinhao.jpg" alt="视频号" width="300" height="350">

