# -*- coding: utf-8 -*-
# @Time    : 2024/7/13 21:21
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : RealTimeLivePortrait
# @FileName: motion_extractor_model.py
import pdb

from .base_model import BaseModel


class AppearanceFeatureExtractorModel(BaseModel):
    """
    AppearanceFeatureExtractorModel
    """

    def __init__(self, **kwargs):
        super(AppearanceFeatureExtractorModel, self).__init__(**kwargs)
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
        output = self.output_process(*preds)
        return output