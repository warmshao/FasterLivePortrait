# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: gradio_live_portrait_pipeline.py
import pdb

import gradio as gr
import cv2
import datetime
import os
import time
from tqdm import tqdm
import subprocess
import numpy as np
from .faster_live_portrait_pipeline import FasterLivePortraitPipeline
from ..utils.utils import video_has_audio
from ..utils.utils import resize_to_limit, prepare_paste_back, get_rotation_matrix, calc_lip_close_ratio, \
    calc_eye_close_ratio, transform_keypoint, concat_feat
from ..utils.crop import crop_image, parse_bbox_from_landmark, crop_image_by_bbox, paste_back
from src.utils import utils
import platform

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


class GradioLivePortraitPipeline(FasterLivePortraitPipeline):
    def __init__(self, cfg, **kwargs):
        super(GradioLivePortraitPipeline, self).__init__(cfg, **kwargs)

    def update_cfg(self, args_user):
        update_ret = False
        for key in args_user:
            if key in self.cfg.infer_params:
                if self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print("update infer cfg {} from {} to {}".format(key, self.cfg.infer_params[key], args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
            elif key in self.cfg.crop_params:
                if self.cfg.crop_params[key] != args_user[key]:
                    update_ret = True
                print("update crop cfg {} from {} to {}".format(key, self.cfg.crop_params[key], args_user[key]))
                self.cfg.crop_params[key] = args_user[key]
            else:
                if key in self.cfg.infer_params and self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print("add {}:{} to infer cfg".format(key, args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
        return update_ret

    def execute_video(
            self,
            input_source_image_path=None,
            input_source_video_path=None,
            input_driving_video_path=None,
            flag_relative_input=True,
            flag_do_crop_input=True,
            flag_remap_input=True,
            driving_multiplier=1.0,
            flag_stitching=True,
            flag_crop_driving_video_input=True,
            flag_video_editing_head_rotation=False,
            flag_is_animal=False,
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            driving_smooth_observation_variance=1e-7,
            tab_selection=None,
    ):
        """ for video driven potrait animation
        """
        if tab_selection == 'Image':
            input_source_path = input_source_image_path
        elif tab_selection == 'Video':
            input_source_path = input_source_video_path
        else:
            input_source_path = input_source_image_path

        if flag_is_animal != self.is_animal:
            self.init_models(is_animal=flag_is_animal)

        if input_source_path is not None and input_driving_video_path is not None:
            args_user = {
                'source': input_source_path,
                'driving': input_driving_video_path,
                'flag_relative_motion': flag_relative_input,
                'flag_do_crop': flag_do_crop_input,
                'flag_pasteback': flag_remap_input,
                'driving_multiplier': driving_multiplier,
                'flag_stitching': flag_stitching,
                'flag_crop_driving_video': flag_crop_driving_video_input,
                'flag_video_editing_head_rotation': flag_video_editing_head_rotation,
                'src_scale': scale,
                'src_vx_ratio': vx_ratio,
                'src_vy_ratio': vy_ratio,
                'dri_scale': scale_crop_driving_video,
                'dri_vx_ratio': vx_ratio_crop_driving_video,
                'dri_vy_ratio': vy_ratio_crop_driving_video,
                'driving_smooth_observation_variance': driving_smooth_observation_variance,
            }
            # update config from user input
            update_ret = self.update_cfg(args_user)
            # video driven animation
            video_path, video_path_concat, total_time = self.run_local(input_driving_video_path, input_source_path,
                                                                       update_ret=update_ret)
            gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
            return video_path, video_path_concat,
        else:
            raise gr.Error("The input source portrait or driving video hasn't been prepared yet 💥!", duration=5)

    def run_local(self, driving_video_path, source_path, **kwargs):
        t00 = time.time()

        if self.source_path != source_path or kwargs.get("update_ret", False):
            # 如果不一样要重新初始化变量
            self.init_vars(**kwargs)
            ret = self.prepare_source(source_path)
            if not ret:
                raise gr.Error(f"Error in processing source:{source_path} 💥!", duration=5)

        vcap = cv2.VideoCapture(driving_video_path)
        fps = int(vcap.get(cv2.CAP_PROP_FPS))
        dframe = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.is_source_video:
            max_frame = min(dframe, len(self.src_imgs))
        else:
            max_frame = dframe
        h, w = self.src_imgs[0].shape[:2]
        save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        # render output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(source_path)}-{os.path.basename(driving_video_path)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(source_path)}-{os.path.basename(driving_video_path)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

        infer_times = []
        for i in tqdm(range(max_frame)):
            ret, frame = vcap.read()
            if not ret:
                break
            t0 = time.time()
            if self.is_source_video:
                dri_crop, out_crop, out_org = self.run(frame, self.src_imgs[i], self.src_infos[i])
            else:
                dri_crop, out_crop, out_org = self.run(frame, self.src_imgs[0], self.src_infos[0])
            infer_times.append(time.time() - t0)
            dri_crop = cv2.resize(dri_crop, (512, 512))
            out_crop = np.concatenate([dri_crop, out_crop], axis=1)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        total_time = time.time() - t00
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

            return vsave_org_path_new, vsave_crop_path_new, total_time
        else:
            return vsave_org_path, vsave_crop_path, total_time

    def execute_image(self, input_eye_ratio: float, input_lip_ratio: float, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        # disposable feature
        f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
            self.prepare_retargeting(input_image, flag_do_crop)

        if input_eye_ratio is None or input_lip_ratio is None:
            raise gr.Error("Invalid ratio input 💥!", duration=5)
        else:
            # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
            combined_eye_ratio_tensor = self.calc_combined_eye_ratio([[input_eye_ratio]], source_lmk_user)
            eyes_delta = self.retarget_eye(x_s_user, combined_eye_ratio_tensor)
            # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
            combined_lip_ratio_tensor = self.calc_combined_lip_ratio([[input_lip_ratio]], source_lmk_user)
            lip_delta = self.retarget_lip(x_s_user, combined_lip_ratio_tensor)
            num_kp = x_s_user.shape[1]
            # default: use x_s
            x_d_new = x_s_user + eyes_delta.reshape(-1, num_kp, 3) + lip_delta.reshape(-1, num_kp, 3)
            # D(W(f_s; x_s, x′_d))
            out = self.model_dict["warping_spade"].predict(f_s_user, x_s_user, x_d_new)
            out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori)
            gr.Info("Run successfully!", duration=2)
            return out, out_to_ori_blend

    def prepare_retargeting(self, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            ######## process source portrait ########
            img_bgr = cv2.imread(input_image, cv2.IMREAD_COLOR)
            img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                      self.cfg.infer_params.source_division)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            src_faces = self.model_dict["face_analysis"].predict(img_bgr)

            if len(src_faces) == 0:
                raise gr.Error("No face detect in image 💥!", duration=5)
            src_faces = src_faces[:1]
            crop_infos = []
            for i in range(len(src_faces)):
                # NOTE: temporarily only pick the first face, to support multiple face in the future
                src_face = src_faces[i]
                lmk = src_face.landmark  # this is the 106 landmarks from insightface
                # crop the face
                ret_dct = crop_image(
                    img_rgb,  # ndarray
                    lmk,  # 106x2 or Nx2
                    dsize=self.cfg.crop_params.src_dsize,
                    scale=self.cfg.crop_params.src_scale,
                    vx_ratio=self.cfg.crop_params.src_vx_ratio,
                    vy_ratio=self.cfg.crop_params.src_vy_ratio,
                )
                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                ret_dct["lmk_crop"] = lmk

                # update a 256x256 version for network input
                ret_dct["img_crop_256x256"] = cv2.resize(
                    ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
                )
                ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize
                crop_infos.append(ret_dct)
            crop_info = crop_infos[0]
            if flag_do_crop:
                I_s = crop_info['img_crop_256x256'].copy()
            else:
                I_s = img_rgb.copy()
            pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(I_s)
            x_s_info = {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "t": t,
                "exp": exp,
                "scale": scale,
                "kp": kp
            }
            R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            ############################################
            f_s_user = self.model_dict["app_feat_extractor"].predict(I_s)
            x_s_user = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
            source_lmk_user = crop_info['lmk_crop']
            crop_M_c2o = crop_info['M_c2o']

            mask_ori = prepare_paste_back(self.mask_crop, crop_info['M_c2o'],
                                          dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            return f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            # when press the clear button, go here
            raise gr.Error("The retargeting input hasn't been prepared yet 💥!", duration=5)
