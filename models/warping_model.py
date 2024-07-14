# -*- coding: utf-8 -*-
# @Time    : 2022/12/30 9:01
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : MetaMoCap
# @FileName: yolo_human_detect_model.py
import pdb

import cv2
import numpy as np

from .base_model import BaseModel


class WarpingModel(BaseModel):
    """
    Warping Model
    """

    def __init__(self, **kwargs):
        super(WarpingModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        print(self.predict_type)

    def input_process(self, *data):
        feature_3d, kp_source, kp_driving = data
        return feature_3d, kp_source, kp_driving

    def output_process(self, *data):
        if self.predict_type == "trt":
            deformation, occlusion_map, out = data
        else:
            occlusion_map, deformation, out = data
        return occlusion_map, deformation, out

    def predict(self, *data):
        feature_3d, kp_source, kp_driving = self.input_process(*data)
        preds = self.predictor.predict(feature_3d, kp_driving, kp_source)
        outputs = self.output_process(*preds)
        return outputs
