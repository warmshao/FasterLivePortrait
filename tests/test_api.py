# -*- coding: utf-8 -*-
# @Time    : 2024/9/14 8:50
# @Project : FasterLivePortrait
# @FileName: test_api.py
import os
import requests
import zipfile
from io import BytesIO
import datetime
import json


def test_with_pickle_animal():
    try:
        data = {
            'flag_is_animal': True,
            'flag_pickle': True,
            'flag_relative_input': True,
            'flag_do_crop_input': True,
            'flag_remap_input': True,
            'driving_multiplier': 1.0,
            'flag_stitching': True,
            'flag_crop_driving_video_input': True,
            'flag_video_editing_head_rotation': False,
            'scale': 2.3,
            'vx_ratio': 0.0,
            'vy_ratio': -0.125,
            'scale_crop_driving_video': 2.2,
            'vx_ratio_crop_driving_video': 0.0,
            'vy_ratio_crop_driving_video': -0.1,
            'driving_smooth_observation_variance': 1e-7
        }
        source_image_path = "./assets/examples/source/s39.jpg"
        driving_pickle_path = "./assets/examples/driving/d8.pkl"

        # 打开文件
        files = {
            'source_image': open(source_image_path, 'rb'),
            'driving_pickle': open(driving_pickle_path, 'rb')
        }

        # 发送 POST 请求
        response = requests.post("http://127.0.0.1:9871/predict/", files=files, data=data)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
            # save files for each request in a different folder
            dt = datetime.datetime.now()
            ts = int(dt.timestamp())
            tgt = f"./results/api_{ts}/"
            os.makedirs(tgt, exist_ok=True)
            zip_ref.extractall(tgt)
            print("Extracted files into", tgt)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")


def test_with_video_animal():
    try:
        data = {
            'flag_is_animal': True,
            'flag_pickle': False,
            'flag_relative_input': True,
            'flag_do_crop_input': True,
            'flag_remap_input': True,
            'driving_multiplier': 1.0,
            'flag_stitching': True,
            'flag_crop_driving_video_input': True,
            'flag_video_editing_head_rotation': False,
            'scale': 2.3,
            'vx_ratio': 0.0,
            'vy_ratio': -0.125,
            'scale_crop_driving_video': 2.2,
            'vx_ratio_crop_driving_video': 0.0,
            'vy_ratio_crop_driving_video': -0.1,
            'driving_smooth_observation_variance': 1e-7
        }
        source_image_path = "./assets/examples/source/s39.jpg"
        driving_video_path = "./assets/examples/driving/d0.mp4"
        files = {
            'source_image': open(source_image_path, 'rb'),
            'driving_video': open(driving_video_path, 'rb')
        }
        response = requests.post("http://127.0.0.1:9871/predict/", files=files, data=data)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
            # save files for each request in a different folder
            dt = datetime.datetime.now()
            ts = int(dt.timestamp())
            tgt = f"./results/api_{ts}/"
            os.makedirs(tgt, exist_ok=True)
            zip_ref.extractall(tgt)
            print("Extracted files into", tgt)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")


def test_with_video_human():
    try:
        data = {
            'flag_is_animal': False,
            'flag_pickle': False,
            'flag_relative_input': True,
            'flag_do_crop_input': True,
            'flag_remap_input': True,
            'driving_multiplier': 1.0,
            'flag_stitching': True,
            'flag_crop_driving_video_input': True,
            'flag_video_editing_head_rotation': False,
            'scale': 2.3,
            'vx_ratio': 0.0,
            'vy_ratio': -0.125,
            'scale_crop_driving_video': 2.2,
            'vx_ratio_crop_driving_video': 0.0,
            'vy_ratio_crop_driving_video': -0.1,
            'driving_smooth_observation_variance': 1e-7
        }
        source_image_path = "./assets/examples/source/s11.jpg"
        driving_video_path = "./assets/examples/driving/d0.mp4"
        files = {
            'source_image': open(source_image_path, 'rb'),
            'driving_video': open(driving_video_path, 'rb')
        }
        response = requests.post("http://127.0.0.1:9871/predict/", files=files, data=data)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
            # save files for each request in a different folder
            dt = datetime.datetime.now()
            ts = int(dt.timestamp())
            tgt = f"./results/api_{ts}/"
            os.makedirs(tgt, exist_ok=True)
            zip_ref.extractall(tgt)
            print("Extracted files into", tgt)

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")


if __name__ == '__main__':
    test_with_video_animal()
    # test_with_pickle_animal()
    # test_with_video_human()
