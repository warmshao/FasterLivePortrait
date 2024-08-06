# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: stitching_model.py

from .base_model import BaseModel
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict


class StitchingModel(BaseModel):
    """
    StitchingModel
    """

    def __init__(self, **kwargs):
        super(StitchingModel, self).__init__(**kwargs)

    def input_process(self, *data):
        input = data[0]
        return input

    def output_process(self, *data):
        return data[0]

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
        data = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(data)
        else:
            preds = self.predictor.predict(data)
        outputs = self.output_process(*preds)
        return outputs
