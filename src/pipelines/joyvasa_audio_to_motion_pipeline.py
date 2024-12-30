# -*- coding: utf-8 -*-
# @Time    : 2024/12/15
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: joyvasa_audio_to_motion_pipeline.py

import math
import pdb

import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import pathlib
import os

from ..models.JoyVASA.dit_talking_head import DitTalkingHead
from ..models.JoyVASA.helper import NullableArgs
from ..utils import utils


class JoyVASAAudio2MotionPipeline:
    """
    JoyVASA 声音生成LivePortrait Motion
    """

    def __init__(self, **kwargs):
        self.device, self.dtype = utils.get_opt_device_dtype()
        # Check if the operating system is Windows
        if os.name == 'nt':
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        motion_model_path = kwargs.get("motion_model_path", "")
        audio_model_path = kwargs.get("audio_model_path", "")
        motion_template_path = kwargs.get("motion_template_path", "")
        model_data = torch.load(motion_model_path, map_location="cpu")
        model_args = NullableArgs(model_data['args'])
        model = DitTalkingHead(motion_feat_dim=model_args.motion_feat_dim,
                               n_motions=model_args.n_motions,
                               n_prev_motions=model_args.n_prev_motions,
                               feature_dim=model_args.feature_dim,
                               audio_model=model_args.audio_model,
                               n_diff_steps=model_args.n_diff_steps,
                               audio_encoder_path=audio_model_path)
        model_data['model'].pop('denoising_net.TE.pe')
        model.load_state_dict(model_data['model'], strict=False)
        model.to(self.device, dtype=self.dtype)
        model.eval()

        # Restore the original PosixPath if it was changed
        if os.name == 'nt':
            pathlib.PosixPath = temp

        self.motion_generator = model
        self.n_motions = model_args.n_motions
        self.n_prev_motions = model_args.n_prev_motions
        self.fps = model_args.fps
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = model_args.pad_mode
        self.use_indicator = model_args.use_indicator
        self.cfg_mode = kwargs.get("cfg_mode", "incremental")
        self.cfg_cond = kwargs.get("cfg_cond", None)
        self.cfg_scale = kwargs.get("cfg_scale", 2.8)
        with open(motion_template_path, 'rb') as fin:
            self.templete_dict = pickle.load(fin)

    @torch.inference_mode()
    def gen_motion_sequence(self, audio_path, **kwargs):
        # preprocess audio
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sample_rate,
                new_freq=16000,
            )
        audio = audio.mean(0).to(self.device, dtype=self.dtype)
        audio_mean, audio_std = torch.mean(audio), torch.std(audio)
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        # crop audio into n_subdivision according to n_motions
        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        if clip_len <= self.n_motions:
            n_subdivision = 1
        else:
            n_subdivision = math.ceil(clip_len / stride)

        # padding
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0
            elif self.pad_mode == 'replicate':
                padding_value = audio[-1]
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

        # generate motions
        coef_list = []
        for i in range(0, n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = torch.ones((1, self.n_motions)).to(self.device) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
            audio_in = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0)

            if i == 0:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                                   indicator=indicator,
                                                                                   cfg_mode=self.cfg_mode,
                                                                                   cfg_cond=self.cfg_cond,
                                                                                   cfg_scale=self.cfg_scale,
                                                                                   dynamic_threshold=0)
            else:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                                   prev_motion_feat.to(self.dtype),
                                                                                   prev_audio_feat.to(self.dtype),
                                                                                   noise.to(self.dtype),
                                                                                   indicator=indicator,
                                                                                   cfg_mode=self.cfg_mode,
                                                                                   cfg_cond=self.cfg_cond,
                                                                                   cfg_scale=self.cfg_scale,
                                                                                   dynamic_threshold=0)
            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]

            motion_coef = motion_feat
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames
            coef_list.append(motion_coef)
            motion_coef = torch.cat(coef_list, dim=1)
            # motion_coef = self.reformat_motion(args, motion_coef)

        motion_coef = motion_coef.squeeze().cpu().numpy().astype(np.float32)
        motion_list = []
        for idx in tqdm(range(motion_coef.shape[0]), total=motion_coef.shape[0]):
            exp = motion_coef[idx][:63] * self.templete_dict["std_exp"] + self.templete_dict["mean_exp"]
            scale = motion_coef[idx][63:64] * (
                    self.templete_dict["max_scale"] - self.templete_dict["min_scale"]) + self.templete_dict[
                        "min_scale"]
            t = motion_coef[idx][64:67] * (self.templete_dict["max_t"] - self.templete_dict["min_t"]) + \
                self.templete_dict["min_t"]
            pitch = motion_coef[idx][67:68] * (
                    self.templete_dict["max_pitch"] - self.templete_dict["min_pitch"]) + self.templete_dict[
                        "min_pitch"]
            yaw = motion_coef[idx][68:69] * (self.templete_dict["max_yaw"] - self.templete_dict["min_yaw"]) + \
                  self.templete_dict["min_yaw"]
            roll = motion_coef[idx][69:70] * (self.templete_dict["max_roll"] - self.templete_dict["min_roll"]) + \
                   self.templete_dict["min_roll"]

            R = utils.get_rotation_matrix(pitch, yaw, roll)
            R = R.reshape(1, 3, 3).astype(np.float32)

            exp = exp.reshape(1, 21, 3).astype(np.float32)
            scale = scale.reshape(1, 1).astype(np.float32)
            t = t.reshape(1, 3).astype(np.float32)
            pitch = pitch.reshape(1, 1).astype(np.float32)
            yaw = yaw.reshape(1, 1).astype(np.float32)
            roll = roll.reshape(1, 1).astype(np.float32)

            motion_list.append({"exp": exp, "scale": scale, "R": R, "t": t, "pitch": pitch, "yaw": yaw, "roll": roll})
        tgt_motion = {'n_frames': motion_coef.shape[0], 'output_fps': self.fps, 'motion': motion_list, 'c_eyes_lst': [],
                      'c_lip_lst': []}
        return tgt_motion
