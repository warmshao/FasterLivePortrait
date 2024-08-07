# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: motion_extractor_model.py
import pdb

import numpy as np

from .base_model import BaseModel
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict
import torch.nn.functional as F


def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        idx_array = np.arange(0, 66)
        pred = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, pred)
        degree = np.sum(pred * idx_array, axis=1) * 3 - 97.5

        return degree

    return pred


class MotionExtractorModel(BaseModel):
    """
    MotionExtractorModel
    """

    def __init__(self, **kwargs):
        super(MotionExtractorModel, self).__init__(**kwargs)
        self.flag_refine_info = kwargs.get("flag_refine_info", True)

    def input_process(self, *data):
        img = data[0].astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[None]

    def output_process(self, *data):
        if self.predict_type == "trt":
            kp, pitch, yaw, roll, t, exp, scale = data
        else:
            pitch, yaw, roll, t, exp, scale, kp = data
        if self.flag_refine_info:
            bs = kp.shape[0]
            pitch = headpose_pred_to_degree(pitch)[:, None]  # Bx1
            yaw = headpose_pred_to_degree(yaw)[:, None]  # Bx1
            roll = headpose_pred_to_degree(roll)[:, None]  # Bx1
            kp = kp.reshape(bs, -1, 3)  # BxNx3
            exp = exp.reshape(bs, -1, 3)  # BxNx3
        return pitch, yaw, roll, t, exp, scale, kp

    def predict_trt(self, *data):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i]
            else:
                feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                      dtype=numpy_to_torch_dtype_dict[inp['dtype']])
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            outs.append(preds_dict[out["name"]].cpu().numpy())
        nvtx.range_pop()
        return outs

    def predict(self, *data):
        img = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(img)
        else:
            preds = self.predictor.predict(img)
        outputs = self.output_process(*preds)
        return outputs
