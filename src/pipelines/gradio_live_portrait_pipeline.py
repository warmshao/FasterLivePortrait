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
import pickle
import numpy as np
from .faster_live_portrait_pipeline import FasterLivePortraitPipeline
from .joyvasa_audio_to_motion_pipeline import JoyVASAAudio2MotionPipeline
from ..utils.utils import video_has_audio
from ..utils.utils import resize_to_limit, prepare_paste_back, get_rotation_matrix, calc_lip_close_ratio, \
    calc_eye_close_ratio, transform_keypoint, concat_feat
from ..utils.crop import crop_image, parse_bbox_from_landmark, crop_image_by_bbox, paste_back, paste_back_pytorch
from src.utils import utils
import platform
import torch
from PIL import Image

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


class GradioLivePortraitPipeline(FasterLivePortraitPipeline):
    def __init__(self, cfg, **kwargs):
        super(GradioLivePortraitPipeline, self).__init__(cfg, **kwargs)
        self.joyvasa_pipe = None

    def execute_video(
            self,
            input_source_image_path=None,
            input_source_video_path=None,
            input_driving_video_path=None,
            input_driving_image_path=None,
            input_driving_pickle_path=None,
            input_driving_audio_path=None,
            flag_relative_input=True,
            flag_do_crop_input=True,
            flag_remap_input=True,
            driving_multiplier=1.0,
            flag_stitching=True,
            flag_crop_driving_video_input=True,
            flag_video_editing_head_rotation=False,
            flag_is_animal=False,
            animation_region="all",
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            driving_smooth_observation_variance=1e-7,
            tab_selection=None,
            v_tab_selection=None,
            cfg_scale=4.0
    ):
        """ for video driven potrait animation
        """
        if tab_selection == 'Video':
            input_source_path = input_source_video_path
        else:
            input_source_path = input_source_image_path

        if v_tab_selection == 'Image':
            input_driving_path = str(input_driving_image_path)
        elif v_tab_selection == 'Pickle':
            input_driving_path = str(input_driving_pickle_path)
        elif v_tab_selection == 'Audio':
            input_driving_path = str(input_driving_audio_path)
        else:
            input_driving_path = str(input_driving_video_path)

        if flag_is_animal != self.is_animal:
            self.init_models(is_animal=flag_is_animal)

        if input_source_path and input_driving_path:
            args_user = {
                'source': input_source_path,
                'driving': input_driving_path,
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
                'animation_region': animation_region,
                'cfg_scale': cfg_scale
            }
            # update config from user input
            update_ret = self.update_cfg(args_user)
            if v_tab_selection == 'Video':
                # video driven animation
                video_path, video_path_concat, total_time = self.run_video_driving(input_driving_path,
                                                                                   input_source_path,
                                                                                   update_ret=update_ret)
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return gr.update(visible=True), video_path, gr.update(visible=True), video_path_concat, gr.update(
                    visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif v_tab_selection == 'Pickle':
                # pickle driven animation
                video_path, video_path_concat, total_time = self.run_pickle_driving(input_driving_path,
                                                                                    input_source_path,
                                                                                    update_ret=update_ret)
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return gr.update(visible=True), video_path, gr.update(visible=True), video_path_concat, gr.update(
                    visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif v_tab_selection == 'Audio':
                # audio driven animation
                video_path, video_path_concat, total_time = self.run_audio_driving(input_driving_path,
                                                                                   input_source_path,
                                                                                   update_ret=update_ret)
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return gr.update(visible=True), video_path, gr.update(visible=True), video_path_concat, gr.update(
                    visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            else:
                # video driven animation
                image_path, image_path_concat, total_time = self.run_image_driving(input_driving_path,
                                                                                   input_source_path,
                                                                                   update_ret=update_ret)
                gr.Info(f"Run successfully! Cost: {total_time} seconds!", duration=3)
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                    visible=False), gr.update(visible=True), image_path, gr.update(
                    visible=True), image_path_concat
        else:
            raise gr.Error("The input source portrait or driving video hasn't been prepared yet 💥!", duration=5)

    def run_image_driving(self, driving_image_path, source_path, **kwargs):
        if self.source_path != source_path or kwargs.get("update_ret", False):
            # 如果不一样要重新初始化变量
            self.init_vars(**kwargs)
            ret = self.prepare_source(source_path)
            if not ret:
                raise gr.Error(f"Error in processing source:{source_path} 💥!", duration=5)

        driving_image = cv2.imread(driving_image_path)
        save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        image_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(source_path)}-{os.path.basename(driving_image_path)}-crop.jpg")
        image_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(source_path)}-{os.path.basename(driving_image_path)}-org.jpg")

        t0 = time.time()
        dri_crop, out_crop, out_org = self.run(driving_image, self.src_imgs[0], self.src_infos[0],
                                               first_frame=True)[:3]

        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_crop_path, out_crop)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_org_path, out_org)
        total_time = time.time() - t0

        return image_org_path, image_crop_path, total_time

    def run_video_driving(self, driving_video_path, source_path, **kwargs):
        t00 = time.time()

        if self.source_path != source_path or kwargs.get("update_ret", False):
            # 如果不一样要重新初始化变量
            self.init_vars(**kwargs)
            ret = self.prepare_source(source_path)
            if not ret:
                raise gr.Error(f"Error in processing source:{source_path} 💥!", duration=5)

        vcap = cv2.VideoCapture(driving_video_path)
        if self.is_source_video:
            duration, fps = utils.get_video_info(self.source_path)
            fps = int(fps)
        else:
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
            first_frame = i == 0
            if self.is_source_video:
                dri_crop, out_crop, out_org = self.run(frame, self.src_imgs[i], self.src_infos[i],
                                                       first_frame=first_frame)[:3]
            else:
                dri_crop, out_crop, out_org = self.run(frame, self.src_imgs[0], self.src_infos[0],
                                                       first_frame=first_frame)[:3]
            if out_crop is None:
                print(f"no face in driving frame:{i}")
                continue
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
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            if self.is_source_video:
                duration, fps = utils.get_video_info(vsave_crop_path)
                subprocess.call(
                    [FFMPEG, "-i", vsave_crop_path, "-i", driving_video_path,
                     "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
                     "-c:a", "aac", "-pix_fmt", "yuv420p",
                     "-shortest",  # 以最短的流为基准
                     "-t", str(duration),  # 设置时长
                     "-r", str(fps),  # 设置帧率
                     vsave_crop_path_new, "-y"])
                subprocess.call(
                    [FFMPEG, "-i", vsave_org_path, "-i", driving_video_path,
                     "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
                     "-c:a", "aac", "-pix_fmt", "yuv420p",
                     "-shortest",  # 以最短的流为基准
                     "-t", str(duration),  # 设置时长
                     "-r", str(fps),  # 设置帧率
                     vsave_org_path_new, "-y"])
            else:
                subprocess.call(
                    [FFMPEG, "-i", vsave_crop_path, "-i", driving_video_path,
                     "-b:v", "10M", "-c:v",
                     "libx264", "-map", "0:v", "-map", "1:a",
                     "-c:a", "aac",
                     "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
                subprocess.call(
                    [FFMPEG, "-i", vsave_org_path, "-i", driving_video_path,
                     "-b:v", "10M", "-c:v",
                     "libx264", "-map", "0:v", "-map", "1:a",
                     "-c:a", "aac",
                     "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            return vsave_org_path_new, vsave_crop_path_new, total_time
        else:
            return vsave_org_path, vsave_crop_path, total_time

    def run_pickle_driving(self, driving_pickle_path, source_path, **kwargs):
        t00 = time.time()

        if self.source_path != source_path or kwargs.get("update_ret", False):
            # 如果不一样要重新初始化变量
            self.init_vars(**kwargs)
            ret = self.prepare_source(source_path)
            if not ret:
                raise gr.Error(f"Error in processing source:{source_path} 💥!", duration=5)

        with open(driving_pickle_path, "rb") as fin:
            dri_motion_infos = pickle.load(fin)

        if self.is_source_video:
            duration, fps = utils.get_video_info(self.source_path)
            fps = int(fps)
        else:
            fps = int(dri_motion_infos["output_fps"])

        motion_lst = dri_motion_infos["motion"]
        c_eyes_lst = dri_motion_infos["c_eyes_lst"] if "c_eyes_lst" in dri_motion_infos else dri_motion_infos[
            "c_d_eyes_lst"]
        c_lip_lst = dri_motion_infos["c_lip_lst"] if "c_lip_lst" in dri_motion_infos else dri_motion_infos[
            "c_d_lip_lst"]
        dframe = len(motion_lst)

        if self.is_source_video:
            max_frame = min(dframe, len(self.src_imgs))
        else:
            max_frame = dframe
        h, w = self.src_imgs[0].shape[:2]
        save_dir = kwargs.get("save_dir", f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)

        # render output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(source_path)}-{os.path.basename(driving_pickle_path)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(source_path)}-{os.path.basename(driving_pickle_path)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

        infer_times = []
        for frame_ind in tqdm(range(max_frame)):
            t0 = time.time()
            first_frame = frame_ind == 0
            dri_motion_info_ = [motion_lst[frame_ind]]
            if c_eyes_lst:
                dri_motion_info_.append(c_eyes_lst[frame_ind])
            else:
                dri_motion_info_.append(None)
            if c_lip_lst:
                dri_motion_info_.append(c_lip_lst[frame_ind])
            else:
                dri_motion_info_.append(None)
            if self.is_source_video:
                out_crop, out_org = self.run_with_pkl(dri_motion_info_, self.src_imgs[frame_ind],
                                                      self.src_infos[frame_ind],
                                                      first_frame=first_frame)[:3]
            else:
                out_crop, out_org = self.run_with_pkl(dri_motion_info_, self.src_imgs[0], self.src_infos[0],
                                                      first_frame=first_frame)[:3]
            if out_crop is None:
                print(f"no face in driving frame:{frame_ind}")
                continue
            infer_times.append(time.time() - t0)
            out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        total_time = time.time() - t00
        vout_crop.release()
        vout_org.release()

        return vsave_org_path, vsave_crop_path, total_time

    def run_audio_driving(self, driving_audio_path, source_path, **kwargs):
        t00 = time.time()

        if self.source_path != source_path or kwargs.get("update_ret", False):
            # 如果不一样要重新初始化变量
            self.init_vars(**kwargs)
            ret = self.prepare_source(source_path)
            if not ret:
                raise gr.Error(f"Error in processing source:{source_path} 💥!", duration=5)
        save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)

        if self.joyvasa_pipe is None:
            self.joyvasa_pipe = JoyVASAAudio2MotionPipeline(motion_model_path=self.cfg.joyvasa_models.motion_model_path,
                                                            audio_model_path=self.cfg.joyvasa_models.audio_model_path,
                                                            motion_template_path=self.cfg.joyvasa_models.motion_template_path,
                                                            cfg_mode=self.cfg.infer_params.cfg_mode,
                                                            cfg_scale=self.cfg.infer_params.cfg_scale
                                                            )
        t01 = time.time()
        dri_motion_infos = self.joyvasa_pipe.gen_motion_sequence(driving_audio_path)
        gr.Info(f"JoyVASA cost time:{time.time() - t01}", duration=2)
        motion_pickle_path = os.path.join(save_dir,
                                          f"{os.path.basename(source_path)}-{os.path.basename(driving_audio_path)}.pkl")
        with open(motion_pickle_path, "wb") as fw:
            pickle.dump(dri_motion_infos, fw)

        vsave_org_path, vsave_crop_path, total_time = self.run_pickle_driving(motion_pickle_path, source_path,
                                                                              save_dir=save_dir)

        vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
        vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"

        duration, fps = utils.get_video_info(vsave_crop_path)
        subprocess.call(
            [FFMPEG, "-i", vsave_crop_path, "-i", driving_audio_path,
             "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac", "-pix_fmt", "yuv420p",
             "-shortest",  # 以最短的流为基准
             "-t", str(duration),  # 设置时长
             "-r", str(fps),  # 设置帧率
             vsave_crop_path_new, "-y"])
        subprocess.call(
            [FFMPEG, "-i", vsave_org_path, "-i", driving_audio_path,
             "-b:v", "10M", "-c:v", "libx264", "-map", "0:v", "-map", "1:a",
             "-c:a", "aac", "-pix_fmt", "yuv420p",
             "-shortest",  # 以最短的流为基准
             "-t", str(duration),  # 设置时长
             "-r", str(fps),  # 设置帧率
             vsave_org_path_new, "-y"])

        return vsave_org_path_new, vsave_crop_path_new, time.time() - t00

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
            img_rgb = torch.from_numpy(img_rgb).to(self.device)
            out_to_ori_blend = paste_back_pytorch(out, crop_M_c2o, img_rgb, mask_ori)
            gr.Info("Run successfully!", duration=2)
            return out.to(dtype=torch.uint8).cpu().numpy(), out_to_ori_blend.to(dtype=torch.uint8).cpu().numpy()

    def prepare_retargeting(self, input_image, flag_do_crop=True):
        """ for single image retargeting
        """
        if input_image is not None:
            ######## process source portrait ########
            img_bgr = cv2.imread(input_image, cv2.IMREAD_COLOR)
            img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                      self.cfg.infer_params.source_division)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if self.is_animal:
                raise gr.Error("Animal Model Not Supported in Face Retarget 💥!", duration=5)
            else:
                src_faces = self.model_dict["face_analysis"].predict(img_bgr)

            if len(src_faces) == 0:
                raise gr.Error("No face detect in image 💥!", duration=5)
            src_faces = src_faces[:1]
            crop_infos = []
            for i in range(len(src_faces)):
                # NOTE: temporarily only pick the first face, to support multiple face in the future
                lmk = src_faces[i]
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
                ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

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
            crop_M_c2o = torch.from_numpy(crop_M_c2o).to(self.device)
            mask_ori = prepare_paste_back(self.mask_crop, crop_info['M_c2o'],
                                          dsize=(img_rgb.shape[1], img_rgb.shape[0]))
            mask_ori = torch.from_numpy(mask_ori).to(self.device).float()
            return f_s_user, x_s_user, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
        else:
            # when press the clear button, go here
            raise gr.Error("The retargeting input hasn't been prepared yet 💥!", duration=5)
