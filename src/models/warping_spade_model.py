# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: warping_spade_model.py
import pdb
import numpy as np
from .base_model import BaseModel
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict


class WarpingSpadeModel(BaseModel):
    """
    WarpingSpade Model
    """

    def __init__(self, **kwargs):
        super(WarpingSpadeModel, self).__init__(**kwargs)

    def input_process(self, *data):
        feature_3d, kp_source, kp_driving = data
        return feature_3d, kp_driving, kp_source

    def output_process(self, *data):
        if self.predict_type != "trt":
            out = torch.from_numpy(data[0]).to(self.device).float()
        else:
            out = data[0]
        out = out.permute(0, 2, 3, 1)
        out = torch.clip(out, 0, 1) * 255
        return out[0]

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
            outs.append(preds_dict[out["name"]].clone())
        nvtx.range_pop()
        return outs

    def predict(self, *data):
        data = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(*data)
        else:
            preds = self.predictor.predict(*data)
        outputs = self.output_process(*preds)
        return outputs
