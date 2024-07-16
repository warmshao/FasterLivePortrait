# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: warping_spade_model.py
import pdb
import numpy as np
from .base_model import BaseModel


class WarpingSpadeModel(BaseModel):
    """
    WarpingSpade Model
    """

    def __init__(self, **kwargs):
        super(WarpingSpadeModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        print(self.predict_type)

    def input_process(self, *data):
        feature_3d, kp_source, kp_driving = data
        return feature_3d, kp_source, kp_driving

    def output_process(self, *data):
        out = data[0]
        out = np.transpose(out, [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out[0]

    def predict(self, *data):
        feature_3d, kp_source, kp_driving = self.input_process(*data)
        preds = self.predictor.predict(feature_3d, kp_driving, kp_source)
        outputs = self.output_process(*preds)
        return outputs
