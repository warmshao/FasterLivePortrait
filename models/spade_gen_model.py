# -*- coding: utf-8 -*-
# @Time    : 2024/7/13 19:40
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : RealTimeLivePortrait
# @FileName: spade_gen_model.py

from .base_model import BaseModel


class SpadeGenModel(BaseModel):
    """
    Warping Model
    """

    def __init__(self, **kwargs):
        super(SpadeGenModel, self).__init__(**kwargs)
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