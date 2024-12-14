## FasterLivePortrait API Usage Guide

### Building the Image
* Decide on an image name, for example `shaoguo/faster_liveportrait_api:v1.0`. Replace the `-t` parameter in the following command with your chosen name.
* Run `docker build -t shaoguo/faster_liveportrait_api:v1.0 -f DockerfileAPI .`

### Running the Image
Ensure that your machine has Nvidia GPU drivers installed. CUDA version should be 12.0 or higher. Two scenarios are described below.

* Running on a Local Machine (typically for self-testing)
  * Modify the image name according to what you defined above.
  * Confirm the service port number, default is `9871`. You can define your own by changing the `SERVER_PORT` environment variable in the command below. Remember to also change `-p 9871:9871` to map the port.
  * Set the model path environment variable `CHECKPOINT_DIR`. If you've previously downloaded FasterLivePortrait's onnx model and converted it to trt, I recommend mapping the model files into the container using `-v`, for example `-v E:\my_projects\FasterLivePortrait\checkpoints:/root/FasterLivePortrait/checkpoints`. This avoids re-downloading the onnx model and doing trt conversion. Otherwise, I will check if `CHECKPOINT_DIR` has models, and if not, I will automatically download (ensure network connectivity) and do trt conversion, which will take considerable time.
  * Run command (note: modify the following command according to your settings):
    ```shell
    docker run -d --gpus=all \
    --name faster_liveportrait_api \
    -v E:\my_projects\FasterLivePortrait\checkpoints:/root/FasterLivePortrait/checkpoints \
    -e CHECKPOINT_DIR=/root/FasterLivePortrait/checkpoints \
    -e SERVER_PORT=9871 \
    -p 9871:9871 \
    --restart=always \
    shaoguo/faster_liveportrait_api:v1.0 \
    /bin/bash
    ```
  * Normal operation should display the following information(docker logs $container_id). The running logs are saved in `/root/FasterLivePortrait/logs/log_run.log`:
    ```shell
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:9871 (Press CTRL+C to quit)
    ```

* Running on Cloud GPU Cluster (production environment)
  * This needs to be configured according to different clusters, but the core is the configuration of docker image and environment variables.
  * Load balancing may need to be set up.

### API Call Testing
Refer to `tests/test_api.py`. The default is the Animal model, but now it also supports the Human model.
The return is a compressed package, by default unzipped to `./results/api_*`. Confirm according to the actual printed log.
* `test_with_video_animal()`, image and video driving. Set `flag_pickle=False`. It will additionally return the driving video's pkl file, which can be called directly next time.
* `test_with_pkl_animal()`, image and pkl driving.
* `test_with_video_human()`, image and video driving under the Human model, set `flag_is_animal=False`