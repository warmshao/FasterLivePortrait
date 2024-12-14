## FasterLivePortrait API使用教程

### 构建镜像

* 确定镜像的名字，比如 `shaoguo/faster_liveportrait_api:v1.0`。确认后替换为下面命令 `-t` 的参数。
* 运行 `docker build -t shaoguo/faster_liveportrait_api:v1.0 -f DockerfileAPI .`

### 运行镜像

请确保你的机器已经装了Nvidia显卡的驱动。CUDA的版本在cuda12.0及以上。以下分两种情况介绍。

* 本地机器运行(一般自己测试使用)
    * 镜像名称根据上面你自己定义的更改。
    * 确认服务的端口号，默认为`9871`，你可以自己定义，更改下面命令里环境变量`SERVER_PORT`。同时要记得更改`-p 9871:9871`,
      将端口映射出来。
    * 设置模型路径环境变量 `CHECKPOINT_DIR`。如果你之前下载过FasterLivePortrait的onnx模型并做过trt的转换，我建议
      是可以通过 `-v`把
      模型文件映射进入容器，比如 `-v E:\my_projects\FasterLivePortrait\checkpoints:/root/FasterLivePortrait/checkpoints`,
      这样就避免重新下载onnx模型和做trt的转换。否则我将会检测`CHECKPOINT_DIR`是否有模型，没有的话，我将自动下载（确保有网络）和做trt的转换，这将耗时比较久的时间。
    * 运行命令(注意你要根据自己的设置更改以下命令的信息）：
  ```shell
    docker run -d --gpus=all \
    --name faster_liveportrait_api \
    -v E:\my_projects\FasterLivePortrait\checkpoints:/root/FasterLivePortrait/checkpoints \
    -e CHECKPOINT_DIR=/root/FasterLivePortrait/checkpoints \
    -e SERVER_PORT=9871 \
    -p 9871:9871 \
    --restart=always \
    shaoguo/faster_liveportrait_api:v1.0
  ```
    * 正常运行应该会显示以下信息(docker logs container_id), 运行的日志保存在`/root/FasterLivePortrait/logs/log_run.log`:
  ```shell
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:9871 (Press CTRL+C to quit)
  ```
* 云端GPU集群运行（生产环境）
    * 这需要根据不同的集群做配置，但核心就是镜像和环境变量的配置。
    * 可能要设置负载均衡。

### API调用测试

可以参考`tests/test_api.py`, 默认是Animal的模型，但现在同时也支持Human的模型了。
返回的是压缩包，默认解压在`./results/api_*`, 根据实际打印出来的日志确认。

* `test_with_video_animal()`, 图像和视频的驱动。设置`flag_pickle=False`。会额外返回driving video的pkl文件，下次可以直接调用。
* `test_with_pkl_animal()`, 图像和pkl的驱动。
* `test_with_video_human()`, Human模型下图像和视频的驱动，设置`flag_is_animal=False`