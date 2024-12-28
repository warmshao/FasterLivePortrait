# -*- coding: utf-8 -*-
# @Time    : 2024/12/28
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: test_gradio_local.py
"""
python tests/test_gradio_local.py \
 --src assets/examples/driving/d13.mp4 \
 --dri assets/examples/driving/d11.mp4 \
 --cfg configs/trt_infer.yaml
"""

import sys
sys.path.append(".")
import os
import argparse
import pdb
import subprocess
import ffmpeg
import cv2
import time
import numpy as np
import os
import datetime
import platform
import pickle
from omegaconf import OmegaConf
from tqdm import tqdm

from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src', required=False, type=str, default="assets/examples/source/s12.jpg",
                        help='source path')
    parser.add_argument('--dri', required=False, type=str, default="assets/examples/driving/d14.mp4",
                        help='driving path')
    parser.add_argument('--cfg', required=False, type=str, default="configs/trt_infer.yaml", help='inference config')
    parser.add_argument('--animal', action='store_true', help='use animal model')
    parser.add_argument('--paste_back', action='store_true', default=False, help='paste back to origin image')
    args, unknown = parser.parse_known_args()

    infer_cfg = OmegaConf.load(args.cfg)
    pipe = GradioLivePortraitPipeline(infer_cfg)
    if args.animal:
        pipe.init_models(is_animal=True)

    dri_ext = os.path.splitext(args.dri)[-1][1:].lower()
    if dri_ext in ["pkl"]:
        out_path, out_path_concat, total_time = pipe.run_pickle_driving(args.dri,
                                                                        args.src,
                                                                        update_ret=True)
    elif dri_ext in ["mp4"]:
        out_path, out_path_concat, total_time = pipe.run_video_driving(args.dri,
                                                                       args.src,
                                                                       update_ret=True)
    elif dri_ext in ["mp3", "wav"]:
        out_path, out_path_concat, total_time = pipe.run_audio_driving(args.dri,
                                                                       args.src,
                                                                       update_ret=True)
    else:
        out_path, out_path_concat, total_time = pipe.run_image_driving(args.dri,
                                                                       args.src,
                                                                       update_ret=True)
    print(out_path, out_path_concat, total_time)
