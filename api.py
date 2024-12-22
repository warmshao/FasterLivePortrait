# -*- coding: utf-8 -*-
# @Time    : 2024/9/13 0:23
# @Project : FasterLivePortrait
# @FileName: api.py
import pdb
import shutil
from typing import Optional, Dict, Any
import io
import os
import subprocess
import uvicorn
import cv2
import time
import numpy as np
import os
import datetime
import platform
import pickle
from tqdm import tqdm
from pydantic import BaseModel
from fastapi import APIRouter, Depends, FastAPI, Request, Response, UploadFile
from fastapi import File, Body, Form
from omegaconf import OmegaConf
from fastapi.responses import StreamingResponse
from zipfile import ZipFile
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio
from src.utils import logger

# model dir
project_dir = os.path.dirname(__file__)
checkpoints_dir = os.environ.get("FLIP_CHECKPOINT_DIR", os.path.join(project_dir, "checkpoints"))
log_dir = os.path.join(project_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
result_dir = os.path.join(project_dir, "results")
os.makedirs(result_dir, exist_ok=True)

logger_f = logger.get_logger("faster_liveportrait_api", log_file=os.path.join(log_dir, "log_run.log"))

app = FastAPI()

global pipe

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


def check_all_checkpoints_exist(infer_cfg):
    """
    check whether all checkpoints exist
    :return:
    """
    ret = True
    for name in infer_cfg.models:
        if not isinstance(infer_cfg.models[name].model_path, str):
            for i in range(len(infer_cfg.models[name].model_path)):
                infer_cfg.models[name].model_path[i] = infer_cfg.models[name].model_path[i].replace("./checkpoints",
                                                                                                    checkpoints_dir)
                if not os.path.exists(infer_cfg.models[name].model_path[i]) and not os.path.exists(
                        infer_cfg.models[name].model_path[i][:-4] + ".onnx"):
                    return False
        else:
            infer_cfg.models[name].model_path = infer_cfg.models[name].model_path.replace("./checkpoints",
                                                                                          checkpoints_dir)
            if not os.path.exists(infer_cfg.models[name].model_path) and not os.path.exists(
                    infer_cfg.models[name].model_path[:-4] + ".onnx"):
                return False
    for name in infer_cfg.animal_models:
        if not isinstance(infer_cfg.animal_models[name].model_path, str):
            for i in range(len(infer_cfg.animal_models[name].model_path)):
                infer_cfg.animal_models[name].model_path[i] = infer_cfg.animal_models[name].model_path[i].replace(
                    "./checkpoints",
                    checkpoints_dir)
                if not os.path.exists(infer_cfg.animal_models[name].model_path[i]) and not os.path.exists(
                        infer_cfg.animal_models[name].model_path[i][:-4] + ".onnx"):
                    return False
        else:
            infer_cfg.animal_models[name].model_path = infer_cfg.animal_models[name].model_path.replace("./checkpoints",
                                                                                                        checkpoints_dir)
            if not os.path.exists(infer_cfg.animal_models[name].model_path) and not os.path.exists(
                    infer_cfg.animal_models[name].model_path[:-4] + ".onnx"):
                return False

    # XPOSE
    xpose_model_path = os.path.join(checkpoints_dir, "liveportrait_animal_onnx/xpose.pth")
    if not os.path.exists(xpose_model_path):
        return False
    embeddings_cache_9_path = os.path.join(checkpoints_dir, "liveportrait_animal_onnx/clip_embedding_9.pkl")
    if not os.path.exists(embeddings_cache_9_path):
        return False
    embeddings_cache_68_path = os.path.join(checkpoints_dir, "liveportrait_animal_onnx/clip_embedding_68.pkl")
    if not os.path.exists(embeddings_cache_68_path):
        return False
    return ret


def convert_onnx_to_trt_models(infer_cfg):
    ret = True
    for name in infer_cfg.models:
        if not isinstance(infer_cfg.models[name].model_path, str):
            for i in range(len(infer_cfg.models[name].model_path)):
                trt_path = infer_cfg.models[name].model_path[i]
                onnx_path = trt_path[:-4] + ".onnx"
                if not os.path.exists(trt_path):
                    convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                    logger_f.info(f"convert onnx model: {onnx_path}")
                    result = subprocess.run(convert_cmd, shell=True, check=True)
                    # 检查结果
                    if result.returncode == 0:
                        logger_f.info(f"convert onnx model: {onnx_path} successful")
                    else:
                        logger_f.error(f"convert onnx model: {onnx_path} failed")
                        return False
        else:
            trt_path = infer_cfg.models[name].model_path
            onnx_path = trt_path[:-4] + ".onnx"
            if not os.path.exists(trt_path):
                convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                logger_f.info(f"convert onnx model: {onnx_path}")
                result = subprocess.run(convert_cmd, shell=True, check=True)
                # 检查结果
                if result.returncode == 0:
                    logger_f.info(f"convert onnx model: {onnx_path} successful")
                else:
                    logger_f.error(f"convert onnx model: {onnx_path} failed")
                    return False

    for name in infer_cfg.animal_models:
        if not isinstance(infer_cfg.animal_models[name].model_path, str):
            for i in range(len(infer_cfg.animal_models[name].model_path)):
                trt_path = infer_cfg.animal_models[name].model_path[i]
                onnx_path = trt_path[:-4] + ".onnx"
                if not os.path.exists(trt_path):
                    convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                    logger_f.info(f"convert onnx model: {onnx_path}")
                    result = subprocess.run(convert_cmd, shell=True, check=True)
                    # 检查结果
                    if result.returncode == 0:
                        logger_f.info(f"convert onnx model: {onnx_path} successful")
                    else:
                        logger_f.error(f"convert onnx model: {onnx_path} failed")
                        return False
        else:
            trt_path = infer_cfg.animal_models[name].model_path
            onnx_path = trt_path[:-4] + ".onnx"
            if not os.path.exists(trt_path):
                convert_cmd = f"python scripts/onnx2trt.py -o {onnx_path}"
                logger_f.info(f"convert onnx model: {onnx_path}")
                result = subprocess.run(convert_cmd, shell=True, check=True)
                # 检查结果
                if result.returncode == 0:
                    logger_f.info(f"convert onnx model: {onnx_path} successful")
                else:
                    logger_f.error(f"convert onnx model: {onnx_path} failed")
                    return False
    return ret


@app.on_event("startup")
async def startup_event():
    global pipe
    # default use trt model
    cfg_file = os.path.join(project_dir, "configs/trt_infer.yaml")
    infer_cfg = OmegaConf.load(cfg_file)
    checkpoints_exist = check_all_checkpoints_exist(infer_cfg)

    # first: download model if not exist
    if not checkpoints_exist:
        download_cmd = f"huggingface-cli download warmshao/FasterLivePortrait --local-dir {checkpoints_dir}"
        logger_f.info(f"download model: {download_cmd}")
        result = subprocess.run(download_cmd, shell=True, check=True)
        # 检查结果
        if result.returncode == 0:
            logger_f.info(f"Download checkpoints to {checkpoints_dir} successful")
        else:
            logger_f.error(f"Download checkpoints to {checkpoints_dir} failed")
            exit(1)
    # second: convert onnx model to trt
    convert_ret = convert_onnx_to_trt_models(infer_cfg)
    if not convert_ret:
        logger_f.error(f"convert onnx model to trt failed")
        exit(1)

    infer_cfg.infer_params.flag_pasteback = True
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=True)


def run_with_video(source_image_path, driving_video_path, save_dir):
    global pipe
    ret = pipe.prepare_source(source_image_path, realtime=False)
    if not ret:
        logger_f.warning(f"no face in {source_image_path}! exit!")
        return
    vcap = cv2.VideoCapture(driving_video_path)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]

    # render output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vsave_crop_path = os.path.join(save_dir,
                                   f"{os.path.basename(source_image_path)}-{os.path.basename(driving_video_path)}-crop.mp4")
    vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
    vsave_org_path = os.path.join(save_dir,
                                  f"{os.path.basename(source_image_path)}-{os.path.basename(driving_video_path)}-org.mp4")
    vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0],
                                                                first_frame=first_frame)
        frame_ind += 1
        if out_crop is None:
            logger_f.warning(f"no face in driving frame:{frame_ind}")
            continue

        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])

        infer_times.append(time.time() - t0)
        # print(time.time() - t0)
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        vout_crop.write(out_crop)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        vout_org.write(out_org)
    vcap.release()
    vout_crop.release()
    vout_org.release()
    if video_has_audio(driving_video_path):
        vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
        subprocess.call(
            [FFMPEG, "-i", vsave_crop_path, "-i", driving_video_path,
             "-b:v", "10M", "-c:v",
             "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac",
             "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
        vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
        subprocess.call(
            [FFMPEG, "-i", vsave_org_path, "-i", driving_video_path,
             "-b:v", "10M", "-c:v",
             "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac",
             "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

        logger_f.info(vsave_crop_path_new)
        logger_f.info(vsave_org_path_new)
    else:
        logger_f.info(vsave_crop_path)
        logger_f.info(vsave_org_path)

    logger_f.info(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))
    # save driving motion to pkl
    template_dct = {
        'n_frames': len(motion_lst),
        'output_fps': fps,
        'motion': motion_lst,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }
    template_pkl_path = os.path.join(save_dir,
                                     f"{os.path.basename(driving_video_path)}.pkl")
    with open(template_pkl_path, "wb") as fw:
        pickle.dump(template_dct, fw)
    logger_f.info(f"save driving motion pkl file at : {template_pkl_path}")


def run_with_pkl(source_image_path, driving_pickle_path, save_dir):
    global pipe
    ret = pipe.prepare_source(source_image_path, realtime=False)
    if not ret:
        logger_f.warning(f"no face in {source_image_path}! exit!")
        return

    with open(driving_pickle_path, "rb") as fin:
        dri_motion_infos = pickle.load(fin)

    fps = int(dri_motion_infos["output_fps"])
    h, w = pipe.src_imgs[0].shape[:2]

    # render output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vsave_crop_path = os.path.join(save_dir,
                                   f"{os.path.basename(source_image_path)}-{os.path.basename(driving_pickle_path)}-crop.mp4")
    vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
    vsave_org_path = os.path.join(save_dir,
                                  f"{os.path.basename(source_image_path)}-{os.path.basename(driving_pickle_path)}-org.mp4")
    vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = dri_motion_infos["motion"]
    c_eyes_lst = dri_motion_infos["c_eyes_lst"] if "c_eyes_lst" in dri_motion_infos else dri_motion_infos[
        "c_d_eyes_lst"]
    c_lip_lst = dri_motion_infos["c_lip_lst"] if "c_lip_lst" in dri_motion_infos else dri_motion_infos["c_d_lip_lst"]

    frame_num = len(motion_lst)
    for frame_ind in tqdm(range(frame_num)):
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_motion_info_ = [motion_lst[frame_ind], c_eyes_lst[frame_ind], c_lip_lst[frame_ind]]
        out_crop, out_org = pipe.run_with_pkl(dri_motion_info_, pipe.src_imgs[0], pipe.src_infos[0],
                                              first_frame=first_frame)
        if out_crop is None:
            logger_f.warning(f"no face in driving frame:{frame_ind}")
            continue

        infer_times.append(time.time() - t0)
        # print(time.time() - t0)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        vout_crop.write(out_crop)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        vout_org.write(out_org)

    vout_crop.release()
    vout_org.release()
    logger_f.info(vsave_crop_path)
    logger_f.info(vsave_org_path)
    logger_f.info(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))


class LivePortraitParams(BaseModel):
    flag_pickle: bool = False
    flag_relative_input: bool = True
    flag_do_crop_input: bool = True
    flag_remap_input: bool = True
    driving_multiplier: float = 1.0
    flag_stitching: bool = True
    flag_crop_driving_video_input: bool = True
    flag_video_editing_head_rotation: bool = False
    flag_is_animal: bool = True
    scale: float = 2.3
    vx_ratio: float = 0.0
    vy_ratio: float = -0.125
    scale_crop_driving_video: float = 2.2
    vx_ratio_crop_driving_video: float = 0.0
    vy_ratio_crop_driving_video: float = -0.1
    driving_smooth_observation_variance: float = 1e-7


@app.post("/predict/")
async def upload_files(
        source_image: Optional[UploadFile] = File(None),
        driving_video: Optional[UploadFile] = File(None),
        driving_pickle: Optional[UploadFile] = File(None),
        flag_is_animal: bool = Form(...),
        flag_pickle: bool = Form(...),
        flag_relative_input: bool = Form(...),
        flag_do_crop_input: bool = Form(...),
        flag_remap_input: bool = Form(...),
        driving_multiplier: float = Form(...),
        flag_stitching: bool = Form(...),
        flag_crop_driving_video_input: bool = Form(...),
        flag_video_editing_head_rotation: bool = Form(...),
        scale: float = Form(...),
        vx_ratio: float = Form(...),
        vy_ratio: float = Form(...),
        scale_crop_driving_video: float = Form(...),
        vx_ratio_crop_driving_video: float = Form(...),
        vy_ratio_crop_driving_video: float = Form(...),
        driving_smooth_observation_variance: float = Form(...)
):
    # 根据传入的表单参数构建 infer_params
    infer_params = LivePortraitParams(
        flag_is_animal=flag_is_animal,
        flag_pickle=flag_pickle,
        flag_relative_input=flag_relative_input,
        flag_do_crop_input=flag_do_crop_input,
        flag_remap_input=flag_remap_input,
        driving_multiplier=driving_multiplier,
        flag_stitching=flag_stitching,
        flag_crop_driving_video_input=flag_crop_driving_video_input,
        flag_video_editing_head_rotation=flag_video_editing_head_rotation,
        scale=scale,
        vx_ratio=vx_ratio,
        vy_ratio=vy_ratio,
        scale_crop_driving_video=scale_crop_driving_video,
        vx_ratio_crop_driving_video=vx_ratio_crop_driving_video,
        vy_ratio_crop_driving_video=vy_ratio_crop_driving_video,
        driving_smooth_observation_variance=driving_smooth_observation_variance
    )

    global pipe
    pipe.init_vars()
    if infer_params.flag_is_animal != pipe.is_animal:
        pipe.init_models(is_animal=infer_params.flag_is_animal)

    args_user = {
        'flag_relative_motion': infer_params.flag_relative_input,
        'flag_do_crop': infer_params.flag_do_crop_input,
        'flag_pasteback': infer_params.flag_remap_input,
        'driving_multiplier': infer_params.driving_multiplier,
        'flag_stitching': infer_params.flag_stitching,
        'flag_crop_driving_video': infer_params.flag_crop_driving_video_input,
        'flag_video_editing_head_rotation': infer_params.flag_video_editing_head_rotation,
        'src_scale': infer_params.scale,
        'src_vx_ratio': infer_params.vx_ratio,
        'src_vy_ratio': infer_params.vy_ratio,
        'dri_scale': infer_params.scale_crop_driving_video,
        'dri_vx_ratio': infer_params.vx_ratio_crop_driving_video,
        'dri_vy_ratio': infer_params.vy_ratio_crop_driving_video,
    }
    # update config from user input
    update_ret = pipe.update_cfg(args_user)

    # 保存 source_image 到指定目录
    temp_dir = os.path.join(result_dir, f"temp-{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
    os.makedirs(temp_dir, exist_ok=True)
    if source_image and source_image.filename:
        source_image_path = os.path.join(temp_dir, source_image.filename)
        with open(source_image_path, "wb") as buffer:
            buffer.write(await source_image.read())  # 将内容写入文件
    else:
        source_image_path = None

    if driving_video and driving_video.filename:
        driving_video_path = os.path.join(temp_dir, driving_video.filename)
        with open(driving_video_path, "wb") as buffer:
            buffer.write(await driving_video.read())  # 将内容写入文件
    else:
        driving_video_path = None

    if driving_pickle and driving_pickle.filename:
        driving_pickle_path = os.path.join(temp_dir, driving_pickle.filename)
        with open(driving_pickle_path, "wb") as buffer:
            buffer.write(await driving_pickle.read())  # 将内容写入文件
    else:
        driving_pickle_path = None

    save_dir = os.path.join(result_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    if infer_params.flag_pickle:
        if source_image_path and driving_pickle_path:
            run_with_pkl(source_image_path, driving_pickle_path, save_dir)
    else:
        if source_image_path and driving_video_path:
            run_with_video(source_image_path, driving_video_path, save_dir)
    # zip all files and return
    # 使用 BytesIO 在内存中创建一个字节流
    zip_buffer = io.BytesIO()

    # 使用 ZipFile 将文件夹内容压缩到 zip_buffer 中
    with ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # 添加文件到 ZIP 文件中
                zip_file.write(file_path, arcname=os.path.relpath(file_path, save_dir))

    # 确保缓冲区指针在开始位置，以便读取整个内容
    zip_buffer.seek(0)
    shutil.rmtree(temp_dir)
    shutil.rmtree(save_dir)
    # 通过 StreamingResponse 返回 zip 文件
    return StreamingResponse(zip_buffer, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=output.zip"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("FLIP_IP", "127.0.0.1"), port=os.environ.get("FLIP_PORT", 9871))
