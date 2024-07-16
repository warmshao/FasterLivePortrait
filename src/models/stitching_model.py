# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: stitching_model.py

from .base_model import BaseModel


class StitchingModel(BaseModel):
    """
    StitchingModel
    """

    def __init__(self, **kwargs):
        super(StitchingModel, self).__init__(**kwargs)
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
