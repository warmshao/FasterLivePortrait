# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: motion_extractor_model.py
import pdb
import numpy as np
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
        img = data[0].astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[None]

    def output_process(self, *data):
        return data[0]

    def predict(self, *data):
        img = self.input_process(*data)
        preds = self.predictor.predict(img)
        output = self.output_process(*preds)
        return output