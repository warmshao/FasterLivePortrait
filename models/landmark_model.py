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


class LandmarkModel(BaseModel):
    """
    landmark Model
    """

    def __init__(self, **kwargs):
        super(LandmarkModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        print(self.predict_type)

    def input_process(self, *data):
        input = data[0]
        return input

    def output_process(self, *data):
        return data[0]

    def predict(self, *data):
        input = self.input_process(*data)
        preds = self.predictor.predict(input)
        outputs = self.output_process(*preds)
        return outputs
